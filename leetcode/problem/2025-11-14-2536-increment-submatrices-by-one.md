---
layout: leetcode-entry
title: "2536. Increment Submatrices by One"
permalink: "/leetcode/problem/2025-11-14-2536-increment-submatrices-by-one/"
leetcode_ui: true
entry_slug: "2025-11-14-2536-increment-submatrices-by-one"
---

[2536. Increment Submatrices by One](https://leetcode.com/problems/increment-submatrices-by-one/description) medium
[blog post](https://leetcode.com/problems/increment-submatrices-by-one/solutions/7348057/kotlin-rust-by-samoylenkodmitry-jmcx/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/14112025-2536-increment-submatrices?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/k_crTRCXw94)

![156a70ab-a694-45e7-9288-a04af0d4500f (1).webp](/assets/leetcode_daily_images/a180d6df.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1173

#### Problem TLDR

Increase matrix by query of rectangles of ones #medium #linesweep

#### Intuition

Expand 1D line sweep pattern to 2D: mark entire rows with +1 for start and -1 to end.

#### Approach

* -1 should be on the next cell
* or, you can mark just 4 cells and compute prefix sum in a separate step
#### Complexity

- Time complexity:
$$O(q*n)$$

- Space complexity:
$$O(n^2)$$

#### Code

```kotlin
// 31ms
    fun rangeAddQueries(n: Int, q: Array<IntArray>): Array<IntArray> {
        val cnt = Array(n) { IntArray(n+1) }
        for (q in q) for (y in q[0]..q[2]) {++cnt[y][q[1]]; --cnt[y][q[3]+1]}
        return Array(n) { y -> var c=0; IntArray(n) { x -> c += cnt[y][x];c}}
    }
```
```rust
// 4ms
    pub fn range_add_queries(n: i32, q: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
        let n = n as usize; let mut m = vec![vec![0; n];n];
        for q in q {
            let (a,b,c,d) = (q[0]as usize,q[1]as usize,q[2]as usize,q[3]as usize);
            m[a][b] += 1; if d+1 < n { m[a][d+1] -= 1}; if c+1 < n { m[c+1][b] -= 1}
            if c+1 < n && d+1 < n { m[c+1][d+1] += 1}
        }
        for row in &mut m { for col in 1..n { row[col] += row[col-1] }}
        for row in 1..n { for col in 0..n { m[row][col] += m[row-1][col] }}; m
    }
```

