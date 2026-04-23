---
layout: leetcode-entry
title: "1931. Painting a Grid With Three Different Colors"
permalink: "/leetcode/problem/2025-05-18-1931-painting-a-grid-with-three-different-colors/"
leetcode_ui: true
entry_slug: "2025-05-18-1931-painting-a-grid-with-three-different-colors"
---

[1931. Painting a Grid With Three Different Colors](https://leetcode.com/problems/painting-a-grid-with-three-different-colors/description) hard
[blog post](https://leetcode.com/problems/painting-a-grid-with-three-different-colors/solutions/6755765/kotlin-rust-by-samoylenkodmitry-pgvg/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/18052025-1931-painting-a-grid-with?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/Q_gz_j8sAY8)
![1.webp](/assets/leetcode_daily_images/ffebf79f.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/992

#### Problem TLDR

Ways to 3-color m*n grid, no adjucents #hard #dp #matrix

#### Intuition

The naive DP works:
* walk all possible ways to color the current column with DFS
* pre-compute all possible columns (at max 5 length)
* make a bitmask to compare with the previous
* cache by the `bitmask` and the `current column`

```j

    // masks
    // 00000
    // 12121
    // 12131     3^5 9*9*3 = 81*3 x 1000 = 100.000 ok

```

Then we can rewrite DFS into iterative, and we only have to keep the `previous` result.
For each mask we can use its indice.

Then the `matrix trick`:
* notice we do `n` operations of the same trasformation `op(X) = Y`
* the operation can be described as a `transition matrix`: `Y = X * M`
* doing it `n` times can be written as `Y = X * M^n`

The transition matrix is as follows:
```j

mask1\mask2    a b c d ...
a              1 x               x = 1 if a & b == 0
b
c                  1 x           x = 1 if c & d == 0
d
.
.
.

```
The starting `X` matrix is an identity matrix `i == j ? 1 : 0`.

The `exponentiation trick` is derived from the arithmetics: `x^n = x^(2 * n/2 + n%2) = x^2 * x^(n/2) * x^(n%2)`

#### Approach

* use 3 bits in the mask to quick check `m & mask == 0` later

#### Complexity

- Time complexity:
$$O(n)$$, consider m small as constant

- Space complexity:
$$O(1)$$

#### Code

```kotlin

// 188ms
    fun colorTheGrid(m: Int, n: Int): Int {
        val M = 1000000007; val masks = ArrayList<Int>()
        fun msk(i: Int, curr: Int, prev: Int) {
            if (i > m) masks += curr else
            for (j in 1..3) if (j != prev) msk(i + 1, (curr shl 3) or (1 shl j), j)
        }
        val dp = HashMap<Pair<Int, Int>, Int>()
        fun dfs(i: Int, mask: Int): Int = if (i == n) 1 else
            dp.getOrPut(i to mask) {
                masks.fold(0) { c, m -> if ((m and mask) == 0) (c + dfs(i + 1, m)) % M else c }
            }
        msk(1, 0, 0); return dfs(0, 0)
    }

```
```kotlin

// 98ms
    fun colorTheGrid(m: Int, n: Int): Int {
        val M = 1000000007; val ms = ArrayList<Int>()
        fun msk(i: Int, curr: Int, p: Int) {
            if (i > m) ms += curr else for (j in 1..3) if (j != p)
                msk(i + 1, (curr shl 3) or (1 shl j), j)
        }
        msk(1, 0, 0); val s = ms.size; var n = n - 1; var res = 0
        var m = Array(s) { a -> IntArray(s) { if ((ms[a] and ms[it]) == 0) 1 else 0 }}
        fun mmul(b: Array<IntArray>) = Array(s) { IntArray(s) }.also { r ->
            for (i in 0..<s) for (j in 0..<s) for (k in 0..<s)
                r[i][j] = (r[i][j] + ((1L * b[i][k] * m[k][j]) % M).toInt()) % M
        }
        var mn = Array(s) { y -> IntArray(s) { if (y == it) 1 else 0 }}
        while (n > 0) { if (n % 2 > 0) mn = mmul(mn); m = mmul(m); n /= 2 }
        for (i in 0..<s) for (j in 0..<s) res = (res + mn[i][j]) % M
        return res
    }

```
```kotlin

// 28ms https://leetcode.com/problems/painting-a-grid-with-three-different-colors/submissions/1637068717
    fun colorTheGrid(m: Int, n: Int): Int {
        val M = 1000000007; val masks = ArrayList<Int>()
        fun msk(i: Int, curr: Int, prev: Int) {
            if (i > m) masks += curr else
            for (j in 1..3) if (j != prev) msk(i + 1, (curr shl 3) or (1 shl j), j)
        }
        msk(1, 0, 0); var dp = IntArray(masks.size) { 1 }; var dp2 = IntArray(masks.size)
        for (i in 1..<n) {
            for (mask in masks.indices) {
                var c = 0
                for (m in masks.indices)
                    if ((masks[m] and masks[mask]) == 0) c = (c + dp[m]) % M
                dp2[mask] = c
            }
            dp = dp2.also { dp2 = dp }
        }
        return dp.fold(0) { r, t -> (r + t) % M }
    }

```
```rust

// 11ms
    pub fn color_the_grid(m: i32, n: i32) -> i32 {
        let M = 1000000007; let mut ms = vec![];
        fn msk(i: i32, c: i32, p: i32, ms: &mut Vec<i32>) {
            if i < 1 { ms.push(c); return }
            for j in 1..4 { if j != p { msk(i - 1, (c << 3) | (1 << j), j, ms) }}
        }; msk(m, 0, 0, &mut ms); let s = ms.len();
        let (mut dp, mut dp2) = (vec![1; s], vec![0; s]);
        for _ in 1..n { for mask in 0..s { let mut c = 0; for m in 0..s {
            if (ms[m] & ms[mask]) == 0 { c = (c + dp[m]) % M }}; dp2[mask] = c
            }; (dp, dp2) = (dp2, dp) }
        dp.into_iter().fold(0, |r, t| (r + t) % M)
    }

```
```c++

// 19ms
    int colorTheGrid(int m, int n) {
        vector<int> ms;
        auto msk = [&](this const auto& msk, int i, int c, int p) -> void {
            if (!i) ms.push_back(c); else for (int j = 1; j < 4; ++j)
            if (j != p) msk(i - 1, (c << 3) | (1 << j), j);
        };
        msk(m, 0, 0);
        int M = 1e9+7, s = size(ms), r = 0; vector<int> dp(s, 1), dp2(s);
        for (int i = 1; i < n; ++i) {
            for (int mask = 0; mask < s; ++mask) {
                int c = 0; for (int m = 0; m < s; ++m)
                    if (!(ms[m] & ms[mask])) c = (c + dp[m]) % M;
                dp2[mask] = c;
            }
            dp.swap(dp2);
        }
        for (int x: dp) r = (r + x) % M; return r;
    }

```

