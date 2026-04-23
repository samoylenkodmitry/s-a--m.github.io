---
layout: leetcode-entry
title: "2661. First Completely Painted Row or Column"
permalink: "/leetcode/problem/2025-01-20-2661-first-completely-painted-row-or-column/"
leetcode_ui: true
entry_slug: "2025-01-20-2661-first-completely-painted-row-or-column"
---

[2661. First Completely Painted Row or Column](https://leetcode.com/problems/first-completely-painted-row-or-column/description/) medium
[blog post](https://leetcode.com/problems/first-completely-painted-row-or-column/solutions/6305619/kotlin-rust-by-samoylenkodmitry-w33a/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/20012025-2661-first-completely-painted?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/DjX05QcIXy8)
![1.webp](/assets/leetcode_daily_images/bd877ae3.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/870

#### Problem TLDR

Index of the first filled row/column in 2D matrix #medium #counting

#### Intuition

Two ways of mapping:
* remember positions (y, x) of nums in matrx, the scan the `arr` and count visited rows/columns
* another way is to remember indices of `arr`, then scan matrix horizontally and vertically to find a minimum of maximum row/column index

#### Approach

* do `size + 1` to simplify indexing
* for the first approach, we can store `y * width + x` instead of pairs

#### Complexity

- Time complexity:
$$O(nm)$$

- Space complexity:
$$O(nm)$$

#### Code

```kotlin

    fun firstCompleteIndex(arr: IntArray, mat: Array<IntArray>): Int {
        val ix = IntArray(arr.size + 1); for (i in arr.indices) ix[arr[i]] = i
        return min(mat[0].indices.minOf {  mat.maxOf { r -> ix[r[it]] }},
                   mat.minOf { r -> mat[0].indices.maxOf { ix[r[it]] }})
    }

```
```rust

    pub fn first_complete_index(arr: Vec<i32>, mat: Vec<Vec<i32>>) -> i32 {
        let (mut ix, m, n) = (vec![0; arr.len() + 1], mat.len(), mat[0].len());
        for i in 0..arr.len() { ix[arr[i] as usize] = i }
        (0..n).map(|x| (0..m).map(|y| ix[mat[y][x] as usize]).max().unwrap()).min().unwrap().min(
        (0..m).map(|y| (0..n).map(|x| ix[mat[y][x] as usize]).max().unwrap()).min().unwrap()) as _
    }

```
```c++

    int firstCompleteIndex(vector<int>& arr, vector<vector<int>>& mat) {
        int m = size(mat), n = size(mat[0]); vector<int> ix(size(arr) + 1), r(m, 0), c(n, 0);
        for (int y = 0; y < m; ++y) for (int x = 0; x < n; ++x) ix[mat[y][x]] = y * n + x;
        for (int i = 0; i < size(arr); ++i)
            if (++r[ix[arr[i]] / n] == n || ++c[ix[arr[i]] % n] == m) return i;
        return size(arr);
    }

```

