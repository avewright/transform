/* Random legal chess board generator — bitboard move gen + walk.
 * Compile: cc -O3 -shared -fPIC -o libboardgen.so boardgen.c
 */
#include "boardgen.h"

#include <string.h>
#include <stdio.h>

enum { WHITE = 0, BLACK = 1 };
enum { PAWN = 1, KNIGHT = 2, BISHOP = 3, ROOK = 4, QUEEN = 5, KING = 6 };

#define RANK_1 0x00000000000000FFULL
#define RANK_2 0x000000000000FF00ULL
#define RANK_3 0x0000000000FF0000ULL
#define RANK_4 0x00000000FF000000ULL
#define RANK_5 0x000000FF00000000ULL
#define RANK_6 0x0000FF0000000000ULL
#define RANK_7 0x00FF000000000000ULL
#define RANK_8 0xFF00000000000000ULL
#define FILE_A 0x0101010101010101ULL
#define FILE_H 0x8080808080808080ULL

#define SQ(f, r) ((r) * 8 + (f))
#define BIT(s) (1ULL << (s))

static inline int lsb(uint64_t b) { return __builtin_ctzll(b); }
static inline int pop_lsb(uint64_t *b) {
    int s = __builtin_ctzll(*b);
    *b &= *b - 1;
    return s;
}
static inline int popcount(uint64_t b) { return __builtin_popcountll(b); }

/* ---------- attack tables ---------- */

static uint64_t KNIGHT_ATT[64];
static uint64_t KING_ATT[64];
static uint64_t PAWN_ATT[2][64];
static int INITED = 0;

static uint64_t sliding_att(int sq, uint64_t occ, const int deltas[][2], int n) {
    uint64_t att = 0;
    int f = sq & 7, r = sq >> 3;
    for (int i = 0; i < n; i++) {
        int df = deltas[i][0], dr = deltas[i][1];
        int ff = f + df, rr = r + dr;
        while (ff >= 0 && ff < 8 && rr >= 0 && rr < 8) {
            int s = SQ(ff, rr);
            att |= BIT(s);
            if (occ & BIT(s)) break;
            ff += df;
            rr += dr;
        }
    }
    return att;
}

static const int BISHOP_D[][2] = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
static const int ROOK_D[][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

static uint64_t bishop_att(int sq, uint64_t occ) {
    return sliding_att(sq, occ, BISHOP_D, 4);
}
static uint64_t rook_att(int sq, uint64_t occ) {
    return sliding_att(sq, occ, ROOK_D, 4);
}
static uint64_t queen_att(int sq, uint64_t occ) {
    return bishop_att(sq, occ) | rook_att(sq, occ);
}

static void init_tables(void) {
    if (INITED) return;
    INITED = 1;
    const int nd[8][2] = {{1, 2}, {2, 1}, {-1, 2}, {-2, 1}, {1, -2}, {2, -1}, {-1, -2}, {-2, -1}};
    const int kd[8][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}, {1, 1}, {1, -1}, {-1, 1}, {-1, -1}};

    for (int sq = 0; sq < 64; sq++) {
        int f = sq & 7, r = sq >> 3;
        uint64_t n = 0, k = 0;
        for (int i = 0; i < 8; i++) {
            int ff = f + nd[i][0], rr = r + nd[i][1];
            if (ff >= 0 && ff < 8 && rr >= 0 && rr < 8) n |= BIT(SQ(ff, rr));
            ff = f + kd[i][0]; rr = r + kd[i][1];
            if (ff >= 0 && ff < 8 && rr >= 0 && rr < 8) k |= BIT(SQ(ff, rr));
        }
        KNIGHT_ATT[sq] = n;
        KING_ATT[sq] = k;

        PAWN_ATT[WHITE][sq] = 0;
        PAWN_ATT[BLACK][sq] = 0;
        if (r < 7) {
            if (f > 0) PAWN_ATT[WHITE][sq] |= BIT(SQ(f - 1, r + 1));
            if (f < 7) PAWN_ATT[WHITE][sq] |= BIT(SQ(f + 1, r + 1));
        }
        if (r > 0) {
            if (f > 0) PAWN_ATT[BLACK][sq] |= BIT(SQ(f - 1, r - 1));
            if (f < 7) PAWN_ATT[BLACK][sq] |= BIT(SQ(f + 1, r - 1));
        }
    }
}

/* ---------- board ---------- */

typedef struct {
    uint64_t bb[2][7]; /* [color][piece], [0] unused */
    uint64_t all[2];
    uint64_t occ;
    uint8_t mailbox[64]; /* piece | color<<3, 0=empty */
    int stm;
    int ep;       /* -1 or square */
    int castling; /* 1=WK 2=WQ 4=BK 8=BQ */
    int halfmove;
    int fullmove;
} Board;

typedef struct {
    uint16_t move; /* from | to<<6 | promo<<12 ; promo 0=none, else piece type */
    uint8_t captured;
    uint8_t prev_ep;
    uint8_t prev_castling;
    uint8_t prev_halfmove;
    uint8_t flags; /* 1=ep, 2=castle, 4=double-push */
} Undo;

#define MOVE_FROM(m) ((m) & 63)
#define MOVE_TO(m) (((m) >> 6) & 63)
#define MOVE_PROMO(m) (((m) >> 12) & 7)
#define MK_MOVE(f, t, p) ((uint16_t)((f) | ((t) << 6) | ((p) << 12)))

static inline int piece_at(const Board *b, int sq) {
    return b->mailbox[sq] & 7;
}
static inline int color_at(const Board *b, int sq) {
    return (b->mailbox[sq] >> 3) & 1;
}

static void put_piece(Board *b, int sq, int color, int pt) {
    uint64_t bit = BIT(sq);
    b->bb[color][pt] |= bit;
    b->all[color] |= bit;
    b->occ |= bit;
    b->mailbox[sq] = (uint8_t)(pt | (color << 3));
}

static void remove_piece(Board *b, int sq) {
    int pc = b->mailbox[sq];
    if (!pc) return;
    int pt = pc & 7, color = pc >> 3;
    uint64_t bit = BIT(sq);
    b->bb[color][pt] &= ~bit;
    b->all[color] &= ~bit;
    b->occ &= ~bit;
    b->mailbox[sq] = 0;
}

static void move_piece(Board *b, int from, int to) {
    int pc = b->mailbox[from];
    int pt = pc & 7, color = pc >> 3;
    uint64_t fb = BIT(from), tb = BIT(to);
    b->bb[color][pt] ^= fb | tb;
    b->all[color] ^= fb | tb;
    b->occ ^= fb | tb;
    b->mailbox[to] = pc;
    b->mailbox[from] = 0;
}

static void clear_board(Board *b) {
    memset(b, 0, sizeof(*b));
    b->ep = -1;
    for (int i = 0; i < 64; i++) b->mailbox[i] = 0;
}

static void set_startpos(Board *b) {
    clear_board(b);
    /* white */
    put_piece(b, SQ(0, 0), WHITE, ROOK);
    put_piece(b, SQ(1, 0), WHITE, KNIGHT);
    put_piece(b, SQ(2, 0), WHITE, BISHOP);
    put_piece(b, SQ(3, 0), WHITE, QUEEN);
    put_piece(b, SQ(4, 0), WHITE, KING);
    put_piece(b, SQ(5, 0), WHITE, BISHOP);
    put_piece(b, SQ(6, 0), WHITE, KNIGHT);
    put_piece(b, SQ(7, 0), WHITE, ROOK);
    for (int f = 0; f < 8; f++) put_piece(b, SQ(f, 1), WHITE, PAWN);
    /* black */
    put_piece(b, SQ(0, 7), BLACK, ROOK);
    put_piece(b, SQ(1, 7), BLACK, KNIGHT);
    put_piece(b, SQ(2, 7), BLACK, BISHOP);
    put_piece(b, SQ(3, 7), BLACK, QUEEN);
    put_piece(b, SQ(4, 7), BLACK, KING);
    put_piece(b, SQ(5, 7), BLACK, BISHOP);
    put_piece(b, SQ(6, 7), BLACK, KNIGHT);
    put_piece(b, SQ(7, 7), BLACK, ROOK);
    for (int f = 0; f < 8; f++) put_piece(b, SQ(f, 6), BLACK, PAWN);
    b->stm = WHITE;
    b->castling = 1 | 2 | 4 | 8;
    b->ep = -1;
    b->halfmove = 0;
    b->fullmove = 1;
}

static int square_attacked(const Board *b, int sq, int by) {
    uint64_t occ = b->occ;
    if (PAWN_ATT[by ^ 1][sq] & b->bb[by][PAWN]) return 1;
    if (KNIGHT_ATT[sq] & b->bb[by][KNIGHT]) return 1;
    if (KING_ATT[sq] & b->bb[by][KING]) return 1;
    uint64_t bishops = b->bb[by][BISHOP] | b->bb[by][QUEEN];
    if (bishop_att(sq, occ) & bishops) return 1;
    uint64_t rooks = b->bb[by][ROOK] | b->bb[by][QUEEN];
    if (rook_att(sq, occ) & rooks) return 1;
    return 0;
}

static int king_sq(const Board *b, int color) {
    return lsb(b->bb[color][KING]);
}

static int in_check(const Board *b, int color) {
    return square_attacked(b, king_sq(b, color), color ^ 1);
}

/* castling rights update masks by square */
static int CASTLE_MASK[64];

static void init_castle_masks(void) {
    for (int i = 0; i < 64; i++) CASTLE_MASK[i] = 1 | 2 | 4 | 8;
    CASTLE_MASK[SQ(4, 0)] &= ~(1 | 2); /* white king */
    CASTLE_MASK[SQ(0, 0)] &= ~2;       /* white a1 */
    CASTLE_MASK[SQ(7, 0)] &= ~1;       /* white h1 */
    CASTLE_MASK[SQ(4, 7)] &= ~(4 | 8);
    CASTLE_MASK[SQ(0, 7)] &= ~8;
    CASTLE_MASK[SQ(7, 7)] &= ~4;
}

static void make_move(Board *b, uint16_t move, Undo *u) {
    int from = MOVE_FROM(move), to = MOVE_TO(move), promo = MOVE_PROMO(move);
    int us = b->stm, them = us ^ 1;
    int pt = piece_at(b, from);

    u->move = move;
    u->captured = b->mailbox[to];
    u->prev_ep = (uint8_t)(b->ep + 1); /* store ep+1 so 0 means none */
    u->prev_castling = (uint8_t)b->castling;
    u->prev_halfmove = (uint8_t)b->halfmove;
    u->flags = 0;

    b->ep = -1;
    b->halfmove++;

    /* en passant capture */
    if (pt == PAWN && to == (int)(u->prev_ep - 1) && (u->prev_ep != 0) && !u->captured) {
        int cap_sq = (us == WHITE) ? to - 8 : to + 8;
        u->captured = b->mailbox[cap_sq];
        u->flags = 1;
        remove_piece(b, cap_sq);
        move_piece(b, from, to);
        b->halfmove = 0;
    } else if (pt == KING && (from == SQ(4, 0) || from == SQ(4, 7)) &&
               (to == from + 2 || to == from - 2)) {
        /* castling */
        u->flags = 2;
        move_piece(b, from, to);
        if (to == from + 2) {
            move_piece(b, from + 3, from + 1); /* kingside rook */
        } else {
            move_piece(b, from - 4, from - 1); /* queenside rook */
        }
    } else {
        if (u->captured) {
            remove_piece(b, to);
            b->halfmove = 0;
        }
        if (pt == PAWN) b->halfmove = 0;
        move_piece(b, from, to);
        if (promo) {
            remove_piece(b, to);
            put_piece(b, to, us, promo);
        }
        /* double push -> set ep */
        if (pt == PAWN && (to - from == 16 || from - to == 16)) {
            b->ep = (from + to) / 2;
            u->flags = 4;
        }
    }

    b->castling &= CASTLE_MASK[from];
    b->castling &= CASTLE_MASK[to];
    b->stm = them;
    if (them == WHITE) b->fullmove++;
}

static void unmake_move(Board *b, const Undo *u) {
    int from = MOVE_FROM(u->move), to = MOVE_TO(u->move), promo = MOVE_PROMO(u->move);
    int them = b->stm, us = them ^ 1;

    b->stm = us;
    if (them == WHITE) b->fullmove--;
    b->ep = (int)u->prev_ep - 1;
    b->castling = u->prev_castling;
    b->halfmove = u->prev_halfmove;

    if (u->flags & 2) {
        /* castle */
        move_piece(b, to, from);
        if (to == from + 2) move_piece(b, from + 1, from + 3);
        else move_piece(b, from - 1, from - 4);
        return;
    }

    if (promo) {
        remove_piece(b, to);
        put_piece(b, to, us, PAWN);
    }
    move_piece(b, to, from);

    if (u->flags & 1) {
        /* ep: captured pawn behind `to` */
        int cap_sq = (us == WHITE) ? to - 8 : to + 8;
        int cpt = u->captured & 7, cc = u->captured >> 3;
        put_piece(b, cap_sq, cc, cpt);
    } else if (u->captured) {
        int cpt = u->captured & 7, cc = u->captured >> 3;
        put_piece(b, to, cc, cpt);
    }
}

/* ---------- move generation ---------- */

#define MAX_MOVES 256

static void add_move(uint16_t *moves, int *n, int from, int to, int promo) {
    moves[(*n)++] = MK_MOVE(from, to, promo);
}

static void add_promos(uint16_t *moves, int *n, int from, int to) {
    add_move(moves, n, from, to, QUEEN);
    add_move(moves, n, from, to, ROOK);
    add_move(moves, n, from, to, BISHOP);
    add_move(moves, n, from, to, KNIGHT);
}

static int gen_pseudo(const Board *b, uint16_t *moves) {
    int n = 0;
    int us = b->stm, them = us ^ 1;
    uint64_t occ = b->occ;
    uint64_t enemies = b->all[them];
    uint64_t empty = ~occ;

    /* pawns */
    uint64_t pawns = b->bb[us][PAWN];
    if (us == WHITE) {
        uint64_t single = (pawns << 8) & empty;
        uint64_t promo = single & RANK_8;
        uint64_t quiet = single & ~RANK_8;
        while (quiet) {
            int to = pop_lsb(&quiet);
            add_move(moves, &n, to - 8, to, 0);
        }
        while (promo) {
            int to = pop_lsb(&promo);
            add_promos(moves, &n, to - 8, to);
        }
        uint64_t dbl = ((single & RANK_3) << 8) & empty;
        while (dbl) {
            int to = pop_lsb(&dbl);
            add_move(moves, &n, to - 16, to, 0);
        }
        uint64_t cap_l = ((pawns & ~FILE_A) << 7) & enemies;
        uint64_t cap_r = ((pawns & ~FILE_H) << 9) & enemies;
        while (cap_l) {
            int to = pop_lsb(&cap_l);
            int from = to - 7;
            if (to >= 56) add_promos(moves, &n, from, to);
            else add_move(moves, &n, from, to, 0);
        }
        while (cap_r) {
            int to = pop_lsb(&cap_r);
            int from = to - 9;
            if (to >= 56) add_promos(moves, &n, from, to);
            else add_move(moves, &n, from, to, 0);
        }
        if (b->ep >= 0) {
            uint64_t froms = PAWN_ATT[BLACK][b->ep] & pawns;
            while (froms) {
                int from = pop_lsb(&froms);
                add_move(moves, &n, from, b->ep, 0);
            }
        }
    } else {
        uint64_t single = (pawns >> 8) & empty;
        uint64_t promo = single & RANK_1;
        uint64_t quiet = single & ~RANK_1;
        while (quiet) {
            int to = pop_lsb(&quiet);
            add_move(moves, &n, to + 8, to, 0);
        }
        while (promo) {
            int to = pop_lsb(&promo);
            add_promos(moves, &n, to + 8, to);
        }
        uint64_t dbl = ((single & RANK_6) >> 8) & empty;
        while (dbl) {
            int to = pop_lsb(&dbl);
            add_move(moves, &n, to + 16, to, 0);
        }
        uint64_t cap_l = ((pawns & ~FILE_H) >> 7) & enemies;
        uint64_t cap_r = ((pawns & ~FILE_A) >> 9) & enemies;
        while (cap_l) {
            int to = pop_lsb(&cap_l);
            int from = to + 7;
            if (to < 8) add_promos(moves, &n, from, to);
            else add_move(moves, &n, from, to, 0);
        }
        while (cap_r) {
            int to = pop_lsb(&cap_r);
            int from = to + 9;
            if (to < 8) add_promos(moves, &n, from, to);
            else add_move(moves, &n, from, to, 0);
        }
        if (b->ep >= 0) {
            uint64_t froms = PAWN_ATT[WHITE][b->ep] & pawns;
            while (froms) {
                int from = pop_lsb(&froms);
                add_move(moves, &n, from, b->ep, 0);
            }
        }
    }

    /* knights */
    uint64_t bb = b->bb[us][KNIGHT];
    while (bb) {
        int from = pop_lsb(&bb);
        uint64_t att = KNIGHT_ATT[from] & ~b->all[us];
        while (att) {
            int to = pop_lsb(&att);
            add_move(moves, &n, from, to, 0);
        }
    }

    /* bishops */
    bb = b->bb[us][BISHOP];
    while (bb) {
        int from = pop_lsb(&bb);
        uint64_t att = bishop_att(from, occ) & ~b->all[us];
        while (att) {
            int to = pop_lsb(&att);
            add_move(moves, &n, from, to, 0);
        }
    }

    /* rooks */
    bb = b->bb[us][ROOK];
    while (bb) {
        int from = pop_lsb(&bb);
        uint64_t att = rook_att(from, occ) & ~b->all[us];
        while (att) {
            int to = pop_lsb(&att);
            add_move(moves, &n, from, to, 0);
        }
    }

    /* queens */
    bb = b->bb[us][QUEEN];
    while (bb) {
        int from = pop_lsb(&bb);
        uint64_t att = queen_att(from, occ) & ~b->all[us];
        while (att) {
            int to = pop_lsb(&att);
            add_move(moves, &n, from, to, 0);
        }
    }

    /* king */
    {
        int from = king_sq(b, us);
        uint64_t att = KING_ATT[from] & ~b->all[us];
        while (att) {
            int to = pop_lsb(&att);
            add_move(moves, &n, from, to, 0);
        }
        /* castling */
        if (!in_check(b, us)) {
            if (us == WHITE) {
                if ((b->castling & 1) && !(occ & (BIT(SQ(5, 0)) | BIT(SQ(6, 0)))) &&
                    !square_attacked(b, SQ(5, 0), them) && !square_attacked(b, SQ(6, 0), them))
                    add_move(moves, &n, SQ(4, 0), SQ(6, 0), 0);
                if ((b->castling & 2) && !(occ & (BIT(SQ(1, 0)) | BIT(SQ(2, 0)) | BIT(SQ(3, 0)))) &&
                    !square_attacked(b, SQ(3, 0), them) && !square_attacked(b, SQ(2, 0), them))
                    add_move(moves, &n, SQ(4, 0), SQ(2, 0), 0);
            } else {
                if ((b->castling & 4) && !(occ & (BIT(SQ(5, 7)) | BIT(SQ(6, 7)))) &&
                    !square_attacked(b, SQ(5, 7), them) && !square_attacked(b, SQ(6, 7), them))
                    add_move(moves, &n, SQ(4, 7), SQ(6, 7), 0);
                if ((b->castling & 8) && !(occ & (BIT(SQ(1, 7)) | BIT(SQ(2, 7)) | BIT(SQ(3, 7)))) &&
                    !square_attacked(b, SQ(3, 7), them) && !square_attacked(b, SQ(2, 7), them))
                    add_move(moves, &n, SQ(4, 7), SQ(2, 7), 0);
            }
        }
    }
    return n;
}

static int gen_legal(Board *b, uint16_t *legal) {
    uint16_t pseudo[MAX_MOVES];
    int pn = gen_pseudo(b, pseudo);
    int n = 0;
    int us = b->stm;
    for (int i = 0; i < pn; i++) {
        Undo u;
        make_move(b, pseudo[i], &u);
        if (!in_check(b, us)) legal[n++] = pseudo[i];
        unmake_move(b, &u);
    }
    return n;
}

/* ---------- FEN ---------- */

static const char PIECE_CHAR[16] = {
    '.', 'P', 'N', 'B', 'R', 'Q', 'K', '.',
    '.', 'p', 'n', 'b', 'r', 'q', 'k', '.'
};

static int board_to_fen(const Board *b, char *out, int cap) {
    char *p = out;
    char *end = out + cap - 1;
    for (int r = 7; r >= 0; r--) {
        int empty = 0;
        for (int f = 0; f < 8; f++) {
            int sq = SQ(f, r);
            int pc = b->mailbox[sq];
            if (!pc) {
                empty++;
            } else {
                if (empty) {
                    if (p >= end) return -1;
                    *p++ = (char)('0' + empty);
                    empty = 0;
                }
                if (p >= end) return -1;
                *p++ = PIECE_CHAR[pc];
            }
        }
        if (empty) {
            if (p >= end) return -1;
            *p++ = (char)('0' + empty);
        }
        if (r) {
            if (p >= end) return -1;
            *p++ = '/';
        }
    }
    if (p + 20 >= end) return -1;
    *p++ = ' ';
    *p++ = b->stm == WHITE ? 'w' : 'b';
    *p++ = ' ';
    int wrote = 0;
    if (b->castling & 1) { *p++ = 'K'; wrote = 1; }
    if (b->castling & 2) { *p++ = 'Q'; wrote = 1; }
    if (b->castling & 4) { *p++ = 'k'; wrote = 1; }
    if (b->castling & 8) { *p++ = 'q'; wrote = 1; }
    if (!wrote) *p++ = '-';
    *p++ = ' ';
    if (b->ep >= 0) {
        *p++ = (char)('a' + (b->ep & 7));
        *p++ = (char)('1' + (b->ep >> 3));
    } else {
        *p++ = '-';
    }
    p += sprintf(p, " %d %d", b->halfmove, b->fullmove);
    *p = 0;
    return (int)(p - out);
}

/* ---------- RNG ---------- */

static uint64_t xorshift64(uint64_t *s) {
    uint64_t x = *s;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *s = x;
    return x;
}

static int rand_int(uint64_t *s, int n) {
    /* unbiased enough for our use */
    return (int)(xorshift64(s) % (uint64_t)n);
}

/* ---------- perft ---------- */

static uint64_t perft_rec(Board *b, int depth) {
    if (depth == 0) return 1;
    uint16_t moves[MAX_MOVES];
    int n = gen_legal(b, moves);
    if (depth == 1) return (uint64_t)n;
    uint64_t nodes = 0;
    for (int i = 0; i < n; i++) {
        Undo u;
        make_move(b, moves[i], &u);
        nodes += perft_rec(b, depth - 1);
        unmake_move(b, &u);
    }
    return nodes;
}

uint64_t rbg_perft(int depth) {
    init_tables();
    init_castle_masks();
    Board b;
    set_startpos(&b);
    return perft_rec(&b, depth);
}

const char *rbg_version(void) {
    return "fast_board 1.0.0";
}

/* ---------- public generate ---------- */

static int walk_once(Board *b, uint64_t *rng, int min_ply, int max_ply, int skip_terminal) {
    set_startpos(b);
    int target = min_ply;
    if (max_ply > min_ply) target += rand_int(rng, max_ply - min_ply + 1);

    for (int ply = 0; ply < target; ply++) {
        uint16_t moves[MAX_MOVES];
        Undo u;
        int n = gen_legal(b, moves);
        if (n == 0) return 0;
        make_move(b, moves[rand_int(rng, n)], &u);
        /* discard undo — we never unmake the walk */
    }
    if (skip_terminal) {
        uint16_t moves[MAX_MOVES];
        if (gen_legal(b, moves) == 0) return 0;
    }
    return 1;
}

int rbg_generate_fens_ex(
    char *out,
    int n,
    int min_ply,
    int max_ply,
    uint64_t seed,
    int skip_terminal,
    int max_retries
) {
    if (!out || n <= 0 || min_ply < 0 || max_ply < min_ply) return RBG_ERR;
    init_tables();
    init_castle_masks();
    if (seed == 0) seed = 0xDEADBEEFCAFEBABEULL;

    Board b;
    uint64_t rng = seed;
    int written = 0;
    if (max_retries <= 0) max_retries = 64;

    for (int i = 0; i < n; i++) {
        int ok = 0;
        for (int t = 0; t < max_retries; t++) {
            if (walk_once(&b, &rng, min_ply, max_ply, skip_terminal)) {
                ok = 1;
                break;
            }
        }
        if (!ok) continue;
        board_to_fen(&b, out + (size_t)written * RBG_FEN_MAX, RBG_FEN_MAX);
        written++;
    }
    return written;
}

int rbg_generate_fens(
    char *out,
    int n,
    int min_ply,
    int max_ply,
    uint64_t seed,
    int skip_terminal
) {
    return rbg_generate_fens_ex(out, n, min_ply, max_ply, seed, skip_terminal, 64);
}
