---
layout: leetcode-entry
title: "1605. Find Valid Matrix Given Row and Column Sums"
permalink: "/leetcode/problem/2024-07-20-1605-find-valid-matrix-given-row-and-column-sums/"
leetcode_ui: true
entry_slug: "2024-07-20-1605-find-valid-matrix-given-row-and-column-sums"
---

[1605. Find Valid Matrix Given Row and Column Sums](https://leetcode.com/problems/find-valid-matrix-given-row-and-column-sums/description/) medium
[blog post](https://leetcode.com/problems/find-valid-matrix-given-row-and-column-sums/solutions/5504663/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/20072024-1605-find-valid-matrix-given?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/2kBUwnSGvsg)
![2024-07-20_10-03_1.webp](/assets/leetcode_daily_images/035a70ee.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/676

#### Problem TLDR

Matrix from rows and cols sums #medium

#### Intuition

Let's try to build such a matrix with our bare hands, pen and paper:

![2024-07-20_09-53.webp](/assets/leetcode_daily_images/7ad77366.webp)

I have noticed some interesting facts about this problem:
* there are several valid matrices, all depend on the numbers you choose first
* you have to choose the minimum between the row and column sums, otherwise the sum became bigger than needed
* you can move row by row or column by column
* the more robust strategy is to take as bigger number as possible first, instead of choosing from some of the lower valid values: you don't have to backtrack then

#### Approach

* Use an array initializer in Kotlin

#### Complexity

- Time complexity:
$$O(nm)$$

- Space complexity:
$$O(nm)$$

#### Code

```kotlin

    fun restoreMatrix(rowSum: IntArray, colSum: IntArray) =
        Array(rowSum.size) { y ->
            IntArray(colSum.size) { x ->
                val v = min(rowSum[y], colSum[x])
                rowSum[y] -= v; colSum[x] -= v; v }}

```
```rust

    pub fn restore_matrix(mut row_sum: Vec<i32>, mut col_sum: Vec<i32>) -> Vec<Vec<i32>> {
        let mut res = vec![vec![0; col_sum.len()]; row_sum.len()];
        for y in 0..res.len() { for x in 0..res[0].len() {
            let v = row_sum[y].min(col_sum[x]);
            row_sum[y] -= v; col_sum[x] -= v; res[y][x] = v
        }}; res
    }

```

