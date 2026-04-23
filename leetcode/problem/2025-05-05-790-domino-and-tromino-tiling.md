---
layout: leetcode-entry
title: "790. Domino and Tromino Tiling"
permalink: "/leetcode/problem/2025-05-05-790-domino-and-tromino-tiling/"
leetcode_ui: true
entry_slug: "2025-05-05-790-domino-and-tromino-tiling"
---

[790. Domino and Tromino Tiling](https://leetcode.com/problems/domino-and-tromino-tiling/description/) meidum
[blog post](https://leetcode.com/problems/domino-and-tromino-tiling/solutions/6715961/kotlin-rust-by-samoylenkodmitry-tzbu/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/05052025-790-domino-and-tromino-tiling?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/A85WqarKfQw)
![1.webp](/assets/leetcode_daily_images/fc3a3a00.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/979

#### Problem TLDR

Ways to fill 2xn board with I,L shapes #medium #dp

#### Intuition

Let's full-search by trying every possible way of placing every domino shape at current position `i` with Depth-First Search. If the final column is filled, count this way as `1`. Result only depends on the current position `i` and the column filled condition of `00` as empty, `01` as bottom filled and `10` as top filled. Can be cached.

Another fun way to optimize the solution is to look at the pattern of the results:

```j
1
1
2
5
11     5 * 2 + 1
24    11 * 2 + 2
53    24 * 2 + 5
117   53 * 2 + 11
258
569
1255
2768

```

#### Approach

* for dp we can either go `i+2` or introduce another filled state `11`
* there is also an O(log(n)) solution by doing matrix^n

```j
    // [ a_n   ]   [ 2 0 1 ] [ a_{n-1} ]
    // [a_{n-1}] = [ 1 0 0 ] [ a_{n-2} ]
    // [a_{n-2}]   [ 0 1 0 ] [ a_{n-3} ]
```

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$, or O(1) for the arithmetic solution

#### Code

```kotlin

// 6ms
    fun numTilings(n: Int): Int {
        val M = 1_000_000_007; val dp = HashMap<Pair<Int, Int>, Int>()
        fun dfs(i: Int, tb: Int): Int =  if (i > n) 0 else if (i == n)
        { if (tb == 0) 1 else 0 } else dp.getOrPut(i to tb) {
            val vertical = if (tb > 0) 0 else dfs(i + 1, 0b00)
            val horizontal = dfs(i + 2, 0b00)
            val trtop = if (tb == 0b10) 0 else dfs(i + 1, 0b10)
            val trbot = if (tb == 0b01) 0 else dfs(i + 1, 0b01)
            (((vertical + horizontal) % M + trtop) % M + trbot) % M
        }
        return dfs(0, 0)
    }

```
```kotlin

// 0ms
    fun numTilings(n: Int): Int {
        var a = 1; var b = 1; var c = 2; val m = 1000000007
        for (i in 3..n) { val t = a; a = b; b = c; c = ((2 * b) % m + t) % m }
        return if (n < 2) 1 else if (n < 3) 2 else c
    }

```
```rust

// 0ms
    pub fn num_tilings(n: i32) -> i32 {
        let (mut a, mut b, mut c, m) = (1, 1, 2, 1000000007);
        for i in 3..=n { (a, b, c) = (b, c, ((2 * c) % m + a) % m) }
        if n < 2 { 1 } else if n < 3 { 2 } else { c }
    }

```
```c++

// 0ms
    int numTilings(int n) {
        int c = 2;
        for (int i = 3, t, a = 1, b = 1, m = 1e9+7; i <= n; ++i)
            t = a, a = b, b = c, c = ((2 * b) % m + t) % m;
        return n < 2 ? 1 : n < 3 ? 2 : c;
    }

```

