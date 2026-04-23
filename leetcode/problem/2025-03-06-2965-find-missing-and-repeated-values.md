---
layout: leetcode-entry
title: "2965. Find Missing and Repeated Values"
permalink: "/leetcode/problem/2025-03-06-2965-find-missing-and-repeated-values/"
leetcode_ui: true
entry_slug: "2025-03-06-2965-find-missing-and-repeated-values"
---

[2965. Find Missing and Repeated Values](https://leetcode.com/problems/find-missing-and-repeated-values/description/) easy
[blog post](https://leetcode.com/problems/find-missing-and-repeated-values/solutions/6503950/kotlin-rust-by-samoylenkodmitry-g9ks/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/06032025-2965-find-missing-and-repeated?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/d4ibinjalCk)
![1.webp](/assets/leetcode_daily_images/004b677d.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/916

#### Problem TLDR

Missing and repeated in 1..n array #medium #math

#### Intuition

The expected sum is n * (n + 1) / 2.

```j

    // allsum = sum + r - m
    //
    // m = sum + r - allsum
    //         or
    // r = allsum - sum + m

```
Other trick is the one of:
* HashSet to find repeated
* mark and modify grid to find repeated
* pure math of squares: sq - sq1 = c1 = m^2 - r^2 = (m - r)(m + r), then divide one equation by another s - s1 = c2 = m - r, c1 / c2 = m + r

#### Approach

* try each way of solving
* if you didn't remember the formula x * (x + 1) * (2 * x + 1) / 6, just calculate it (1..n).sumOf { it ^ 2 }

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$, or O(1) for math and mark solutions

#### Code

```kotlin

    fun findMissingAndRepeatedValues(grid: Array<IntArray>) = {
        val all = grid.map { it.asList() }.flatten()
        val m = (1..all.size) - all
        listOf(all.sum() - all.size * (all.size + 1) / 2 + m[0]) + m
    }()

```
```kotlin(mark)

    fun findMissingAndRepeatedValues(grid: Array<IntArray>): IntArray {
        val n = grid.size; val sum = n * n * (n * n + 1) / 2
        val allsum = (0..<n * n).sumOf { grid[it / n][it % n] }
        val i = (0..<n * n).find {
            val v = grid[it / n][it % n];
            val vy = (abs(v) - 1) / n; val vx = (abs(v) - 1) % n
            val u = grid[vy][vx]; grid[vy][vx] *= -1; u < 0 }!!
        val r = abs(grid[i / n][i % n])
        val m = sum + r - allsum
        return intArrayOf(r, m)
    }

```
```rust

    pub fn find_missing_and_repeated_values(grid: Vec<Vec<i32>>) -> Vec<i32> {
        let n = (grid.len() * grid.len()) as i64;
        let (s, sq) = grid.iter().flatten().fold((0, 0), |r, &v| (r.0 + v as i64, r.1 + (v * v) as i64));
        let (c1, c2) = (s - n * (n + 1) / 2, sq - n * (n + 1) * (2 * n + 1) / 6);
        vec![(c2 / c1 + c1) as i32 / 2, (c2 / c1 - c1) as i32 / 2]
    }

```
```c++

    vector<int> findMissingAndRepeatedValues(vector<vector<int>>& g) {
        int r, n =g.size(), s = 0, e = n * n * (n * n + 1) / 2, v[2501] = {};
        for (auto& R: g) for (int x: R) s += x, v[x]++ > 0 ? r = x : 0;
        return {r , e - s + r};
    }

```

