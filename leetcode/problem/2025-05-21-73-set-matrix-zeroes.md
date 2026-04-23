---
layout: leetcode-entry
title: "73. Set Matrix Zeroes"
permalink: "/leetcode/problem/2025-05-21-73-set-matrix-zeroes/"
leetcode_ui: true
entry_slug: "2025-05-21-73-set-matrix-zeroes"
---

[73. Set Matrix Zeroes](https://leetcode.com/problems/set-matrix-zeroes/description/) medium
[blog post](https://leetcode.com/problems/set-matrix-zeroes/solutions/6766228/kotlin-rust-by-samoylenkodmitry-9jpa/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/21052025-73-set-matrix-zeroes?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/bjNHxDU95sQ)
![1.webp](/assets/leetcode_daily_images/c83948db.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/995

#### Problem TLDR

Zero-fy rows and columns #medium

#### Intuition

Do what is asked.
In-place: use first row and column

#### Approach

* don't fall into trap of overriding the good rows
* minimize iterations by checking `m[0][x] || m[y][0]`
* check separately if you need to zero-fy the first row and column

#### Complexity

- Time complexity:
$$O(mn)$$

- Space complexity:
$$O(n)$$ or O(1)

#### Code

```kotlin

// 16ms
    fun setZeroes(m: Array<IntArray>): Unit {
        val zc = m[0].indices.filter { x -> m.any { it[x] == 0 }}
        for (r in m) if (0 in r) r.fill(0)
        for (x in zc) for (r in m) r[x] = 0
    }

```
```kotlin

// 1ms
    fun setZeroes(m: Array<IntArray>): Unit {
        var fr = false; var fc = false; val c = m[0]
        for ((y, r) in m.withIndex()) for (x in r.indices) if (r[x] == 0)
            { if (y == 0) fr = true; if (x == 0) fc = true; c[x] = 0; r[0] = 0 }
        for (y in 1..<m.size) for (x in 1..<c.size)
            if (c[x] == 0 || m[y][0] == 0) m[y][x] = 0
        if (fc) for (r in m) r[0] = 0; if (fr) for (x in c.indices) c[x] = 0
    }

```
```rust

// 0ms
    pub fn set_zeroes(m: &mut Vec<Vec<i32>>) {
        let (mut fr, mut fc) = (false, false);
        for y in 0..m.len() { for x in 0..m[0].len() { if m[y][x] == 0 {
            fr |= y == 0; fc |= x == 0; m[0][x] = 0; m[y][0] = 0
        }}}
        for y in 1..m.len() { for x in 1..m[0].len() {
            if m[0][x] == 0 || m[y][0] == 0 { m[y][x] = 0 }
        }}
        if fc { for y in 0..m.len() { m[y][0] = 0 } }
        if fr { m[0][..].fill(0) }
    }

```
```c++

// 0ms
    void setZeroes(vector<vector<int>>& m) {
        int fr = 0, fc = 0; vector<int>& c = m[0];
        for (int y = 0; y < size(m); ++y) for (int x = 0; x < size(c); ++x) if (!m[y][x])
            fr |= y == 0, fc |= x == 0, c[x] = 0, m[y][0] = 0;
        for (int y = 1; y < size(m); ++y) for (int x = 1; x < size(c); ++x) if (!c[x] || !m[y][0]) m[y][x] = 0;
        if (fc) for (int y = 0; y < size(m); ++y) m[y][0] = 0;
        if (fr) for (int x = 0; x < size(c); ++x) c[x] = 0;
    }

```

