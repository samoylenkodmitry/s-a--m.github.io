---
layout: leetcode-entry
title: "1277. Count Square Submatrices with All Ones"
permalink: "/leetcode/problem/2025-08-20-1277-count-square-submatrices-with-all-ones/"
leetcode_ui: true
entry_slug: "2025-08-20-1277-count-square-submatrices-with-all-ones"
---

[1277. Count Square Submatrices with All Ones](https://leetcode.com/problems/count-square-submatrices-with-all-ones/description/) medium
[blog post](https://leetcode.com/problems/count-square-submatrices-with-all-ones/solutions/7102623/kotlin-rust-by-samoylenkodmitry-zc21/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/20082025-1277-count-square-submatrices?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/_GKdBiHobIE)

![1.webp](/assets/leetcode_daily_images/0dff86b6.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1087

#### Problem TLDR

Count square islands of 1 #medium #dp

#### Intuition

Reuse the previous row calculations:
* look diagonal up-left
* look up
* count left

#### Approach

* try to write without `if`s
* reuse the input (not in production or in interview)

#### Complexity

- Time complexity:
$$O(nm)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

// 21ms
    fun countSquares(m: Array<IntArray>) = with(m) {
        for (y in 1..<size) for (x in 1..<m[0].size)
            m[y][x] *= 1 + min(m[y][x-1], min(m[y-1][x-1], m[y-1][x]))
        sumOf { it.sum() }
    }

```
```rust

// 0ms
    pub fn count_squares(mut m: Vec<Vec<i32>>) -> i32 {
        for y in 1..m.len() { for x in 1..m[0].len() {
            m[y][x] *= 1 + m[y][x-1].min(m[y-1][x-1]).min(m[y-1][x])
        }} m.iter().flatten().sum()
    }

```
```c++

// 0ms
    int countSquares(vector<vector<int>>& m) {
        int r = 0;
        for (int y = 0; y < size(m); ++y) for (int x = 0; x < size(m[0]); ++x)
        r += m[y][x] *= 1 + (y&&x ? min(m[y-1][x-1], min(m[y][x-1], m[y-1][x])): 0);
        return r;
    }

```
```python

// 85ms
    def countSquares(_, m: List[List[int]]):
        for y in range(len(m)):
            for x in range(len(m[0])):
                m[y][x] *= 1 + (y and x and min(m[y-1][x-1], m[y][x-1], m[y-1][x]))
        return sum(map(sum, m))

```

