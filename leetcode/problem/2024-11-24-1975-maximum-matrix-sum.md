---
layout: leetcode-entry
title: "1975. Maximum Matrix Sum"
permalink: "/leetcode/problem/2024-11-24-1975-maximum-matrix-sum/"
leetcode_ui: true
entry_slug: "2024-11-24-1975-maximum-matrix-sum"
---

[1975. Maximum Matrix Sum](https://leetcode.com/problems/maximum-matrix-sum/description/) medium
[blog post](https://leetcode.com/problems/maximum-matrix-sum/solutions/6077947/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/24112024-1975-maximum-matrix-sum?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/91zS-ylJXMs)
[deep-dive](https://notebooklm.google.com/notebook/46446d6a-bc87-4b94-9be6-9be7d582169c/audio)
![1.webp](/assets/leetcode_daily_images/f65a1205.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/810

#### Problem TLDR

Max sum of 2D matrix after multiply by -1 adjacent cells #medium

#### Intuition

This problem is a brainteaser: you must observe how this multiplication by `-1` of adjacent cells works. It works like that:
* Every negative sign can be moved anywhere
* Even negative signs all cancel out
* Odd negative signs leave only a single negative cell

Peek at the smallest value to subtract.

#### Approach

* Imagine the moves, make conclusions
* Can you brute force the multiplication of cells without these simplifications?

#### Complexity

- Time complexity:
$$O(n^2)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun maxMatrixSum(matrix: Array<IntArray>): Long {
        var cnt = 0; var min = Int.MAX_VALUE
        return matrix.sumOf { r -> r.sumOf {
            min = min(min, abs(it))
            if (it < 0) cnt++
            abs(it).toLong()
        }} - 2 * min * (cnt and 1)
    }

```
```rust

    pub fn max_matrix_sum(matrix: Vec<Vec<i32>>) -> i64 {
        let (mut cnt, mut min) = (0, i64::MAX);
        matrix.iter().map(|r|
            r.iter().map(|&v| {
                let a = v.abs() as i64;
                min = min.min(a); if (v < 0) { cnt += 1 }; a
            }).sum::<i64>()
        ).sum::<i64>() - 2 * min * (cnt & 1)
    }

```
```c++

    long long maxMatrixSum(vector<vector<int>>& matrix) {
        int cnt = 0, m = INT_MAX; long long res = 0;
        for (int y = 0; y < matrix.size(); ++y)
            for (int x = 0; x < matrix[0].size(); ++x) {
                m = min(m, abs(matrix[y][x]));
                if (matrix[y][x] < 0) cnt++;
                res += abs(matrix[y][x]);
            }
        return res - 2 * m * (cnt & 1);
    }

```

