---
layout: leetcode-entry
title: "Toeplitz Matrix"
permalink: "/leetcode/problem/2022-10-31-toeplitz-matrix/"
leetcode_ui: true
entry_slug: "2022-10-31-toeplitz-matrix"
---

[https://leetcode.com/problems/toeplitz-matrix/](https://leetcode.com/problems/toeplitz-matrix/) easy

Solution [kotlin]

```kotlin

    fun isToeplitzMatrix(matrix: Array<IntArray>): Boolean =
        matrix
        .asSequence()
        .windowed(2)
        .all { (prev, curr) -> prev.dropLast(1) == curr.drop(1) }

```

Explanation:
just compare adjacent rows, they must have an equal elements except first and last
