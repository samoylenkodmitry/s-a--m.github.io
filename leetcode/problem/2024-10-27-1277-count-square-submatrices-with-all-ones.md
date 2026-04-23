---
layout: leetcode-entry
title: "1277. Count Square Submatrices with All Ones"
permalink: "/leetcode/problem/2024-10-27-1277-count-square-submatrices-with-all-ones/"
leetcode_ui: true
entry_slug: "2024-10-27-1277-count-square-submatrices-with-all-ones"
---

[1277. Count Square Submatrices with All Ones](https://leetcode.com/problems/count-square-submatrices-with-all-ones/description/) medium
[blog post](https://leetcode.com/problems/count-square-submatrices-with-all-ones/solutions/5974085/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/27102024-1277-count-square-submatrices?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/Gj8Up_J5b9Q)
[deep-dive](https://notebooklm.google.com/notebook/38118315-ea9f-40e9-9280-7db675a7287d/audio)
![1.webp](/assets/leetcode_daily_images/b0404eea.webp)

#### Problem TLDR

Count `1`-filled squares in 2D matrix #medium #dynamic_programming

#### Intuition

I failed this one: was in the wrong direction trying to solve with histogram monotonic stack. It didn't work out.

Solution from other people: `dp[y][x]` is the maximum possible size of the filled square ended with a bottom-right (y,x) corner.
By coincidence and pure logic, the size of the square is equal to the number of inside squares with this shared corner in common.

#### Approach

* my personal note: after burning in a one direction for about ~30 minutes it worth to stop hitting the wall to save brain power to grasp others' working solution
* do not do the array modifying trick on the interview without permission, and don't do ever in a production code

#### Complexity

- Time complexity:
$$O(nm)$$

- Space complexity:
$$O(nm)$$ or O(1)

#### Code

```kotlin

    fun countSquares(matrix: Array<IntArray>) =
        matrix.withIndex().sumOf { (y, r) ->
            r.withIndex().sumOf { (x, v) ->
                (v + v * minOf(
                    if (x > 0 && y > 0) matrix[y - 1][x - 1] else 0,
                    if (y > 0) matrix[y - 1][x] else 0,
                    if (x > 0) r[x - 1] else 0
                )).also { r[x] = it }}}

```
```rust

    pub fn count_squares(mut matrix: Vec<Vec<i32>>) -> i32 {
        (0..matrix.len()).map(|y| (0..matrix[0].len()).map(|x| {
            let r = matrix[y][x] * (1 +
                (if x > 0 && y > 0 { matrix[y - 1][x - 1] } else { 0 })
                .min(if y > 0 { matrix[y - 1][x] } else { 0 })
                .min(if x > 0 { matrix[y][x - 1] } else { 0 }));
            matrix[y][x] = r; r
        }).sum::<i32>()).sum()
    }

```
```c++

    int countSquares(vector<vector<int>>& matrix) {
        int res = 0;
        for (int y = 0; y < matrix.size(); ++y)
            for (int x = 0; x < matrix[0].size(); ++x)
                res += (matrix[y][x] *= 1 + min(
                    x > 0 && y > 0 ? matrix[y - 1][x - 1] : 0,
                    min(y > 0 ? matrix[y - 1][x] : 0,
                    x > 0 ? matrix[y][x - 1] : 0)));
        return res;
    }

```

