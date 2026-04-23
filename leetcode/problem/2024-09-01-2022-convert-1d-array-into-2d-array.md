---
layout: leetcode-entry
title: "2022. Convert 1D Array Into 2D Array"
permalink: "/leetcode/problem/2024-09-01-2022-convert-1d-array-into-2d-array/"
leetcode_ui: true
entry_slug: "2024-09-01-2022-convert-1d-array-into-2d-array"
---

[2022. Convert 1D Array Into 2D Array](https://leetcode.com/problems/convert-1d-array-into-2d-array/description/) easy
[blog post](https://leetcode.com/problems/convert-1d-array-into-2d-array/solutions/5719190/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/01092024-2022-convert-1d-array-into?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/2t8cOCmN1JU)
![1.webp](/assets/leetcode_daily_images/1224165e.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/721

#### Problem TLDR

1D to 2D mxn array #easy

#### Intuition

There are many ways to do this:
* `for y in 0..m for x in 0..n` loop
* `for num in original` loop
* using pointer arithmetics in C-like languages (loop unroll and SIMD)
* using Array constructors in Kotlin
* using iterators and `chunks`

#### Approach

* pay attention to the description, we also have to check that size is exactly `m x n`

#### Complexity

- Time complexity:
$$O(n x m)$$

- Space complexity:
$$O(n x m)$$

#### Code

```kotlin

    fun construct2DArray(original: IntArray, m: Int, n: Int) =
        if (original.size != n * m) listOf()
        else original.asList().chunked(n)

```
```rust

    pub fn construct2_d_array(original: Vec<i32>, m: i32, n: i32) -> Vec<Vec<i32>> {
        if original.len() as i32 != m * n { vec![] } else
        { original.chunks_exact(n as usize).map(|r| r.to_vec()).collect() }
    }

```
```c++

    vector<vector<int>> construct2DArray(vector<int>& original, int m, int n) {
        if (original.size() != m * n) return {};
        std::vector<std::vector<int>> result; result.reserve(m);
        const int* dataPtr = original.data();
        for (int i = 0; i < m; ++i)
            result.emplace_back(dataPtr + i * n, dataPtr + i * n + n);
        return result;
    }

```

