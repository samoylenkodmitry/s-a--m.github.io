---
layout: leetcode-entry
title: "3337. Total Characters in String After Transformations II"
permalink: "/leetcode/problem/2025-05-14-3337-total-characters-in-string-after-transformations-ii/"
leetcode_ui: true
entry_slug: "2025-05-14-3337-total-characters-in-string-after-transformations-ii"
---

[3337. Total Characters in String After Transformations II](https://leetcode.com/problems/total-characters-in-string-after-transformations-ii/description) hard
[blog post](https://leetcode.com/problems/total-characters-in-string-after-transformations-ii/solutions/6743063/kotlin-rust-by-samoylenkodmitry-dikn/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/14052025-3337-total-characters-in?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/KnhsmmGHN4c)
![1.webp](/assets/leetcode_daily_images/6056754c.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/988

#### Problem TLDR

`t` steps to convert char into next nums[c] chars #hard #dp #math

#### Intuition

Didn't solve.
Irrelevant chain of thoughts:

```j

    // a -> bc                                        t = 1
    //      b -> cd
    //      c -> de    1c 2d 1e                       t = 2
    //           c -> de
    //          2d -> ef ef
    //           e -> fg       1d 3e 3f 1g            t = 3
    //                d -> ef
    //               3e -> fg fg fg
    //               3f -> gh gh gh
    //                g -> hi         1e 4f 6g 4h 1i   t = 4
    //                                                 ...
    //                                                 t = 26
    //
    // exponentiation...
    // t = 492153482    /26 = 18 928 980.0769

    // 1hr hint "Model the problem as a matrix multiplication problem." lol"

```

How growth law described by matrix:

```j

from\to
       a b c d e f g .. z
a        1 1                  nums[a] = 2
b          1 1 1 1            nums[b] = 4
c            1                nums[c] = 1
d              1 1 1          nums[d] = 3
e
..
z

```

Now, by applying the matrix into initial frequency we will make a single step: `f = f x M`.
To make `t` steps: `f_t = f x M^t`.

The exponentiation trick is from math: `a^t = a^(2 * t/2) + a^(2 * t%2)`

#### Approach

* what's missing: matrix trick for dp

#### Complexity

- Time complexity:
$$O(s + log(t))$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

// 701ms
    fun lengthAfterTransformations(s: String, t: Int, nums: List<Int>): Int {
        var f = LongArray(26); for (c in s) ++f[c - 'a']; val M = 1000000007L
        var m = Array(26) { c -> LongArray(26) }; var t = t; var res = 0L
        for (i in 0..25) for (j in i + 1..<i + nums[i] + 1) m[i][j % 26] = 1
        fun matMul(m1: Array<LongArray>): Array<LongArray> {
            val res = Array(26) { LongArray(26)}
            for (i in 0..25) for (j in 0..25) for (k in 0..25)
                res[i][j] = (res[i][j] + (m1[i][k] * m[k][j]) % M + M) % M
            return res
        }
        var mt = Array(26) { y -> LongArray(26) { x -> if (x == y) 1 else 0 }}
        while (t > 0) { if (t % 2 > 0) mt = matMul(mt); m = matMul(m); t /= 2 }
        for (j in 0..25) for (k in 0..25) res = (res + (f[k] * mt[k][j]) % M) % M
        return res.toInt()
    }

```
```rust

// 24ms
    pub fn length_after_transformations(s: String, mut t: i32, n: Vec<i32>) -> i32 {
        const M: u64 = 1_000_000_007; let (mut f, mut m) = ([0u64; 26], [[0u64; 26]; 26]);
        for b in s.bytes() { f[(b - b'a') as usize] += 1 }
        for i in 0..26 { for j in 1..=n[i] as usize { m[i][(i + j) % 26] = 1 }}
        let mut e = [[0u64; 26]; 26]; for i in 0..26 { e[i][i] = 1 }
        let mul = |a: &[[u64; 26]; 26], b: &[[u64; 26]; 26]| {
            let mut c = [[0u64; 26]; 26];
            for i in 0..26 { for k in 0..26 { let v = a[i][k];
                if v != 0 { for j in 0..26 { c[i][j] = (c[i][j] + v * b[k][j]) % M; }}
            }}; c
        };
        while t > 0 { if t & 1 == 1 { e = mul(&e, &m) }; m = mul(&m, &m); t >>= 1 }
        let mut r = 0u64; for i in 0..26 { for j in 0..26 { r = (r + f[i] * e[i][j]) % M }}
        r as _
    }

```
```c++

// 111ms
    int lengthAfterTransformations(string s, int t, vector<int>& n) {
        long long f[26] = {}, m[26][26] = {}, e[26][26] = {}, c[26][26], r = 0;
        const long long M = 1000000007; for (auto b : s) f[b - 'a']++;
        for (int i = 0; i < 26; i++) for (int j = 1; j <= n[i]; j++) m[i][(i + j) % 26] = 1;
        for (int i = 0; i < 26; i++) e[i][i] = 1;
        auto mul = [&](long long A[26][26], long long B[26][26]) {
            memset(c, 0, sizeof c);
            for (int i = 0; i < 26; i++) for (int k = 0; k < 26; k++) if (A[i][k])
                for (int j = 0; j < 26; j++) c[i][j] = (c[i][j] + A[i][k] * B[k][j]) % M;
            memcpy(A, c, sizeof c);
        };
        while (t) { if (t & 1) mul(e, m); mul(m, m); t >>= 1; }
        for (int i = 0; i < 26; i++) for (int j = 0; j < 26; j++) r = (r + f[i] * e[i][j]) % M;
        return r;
    }

```

