---
layout: leetcode-entry
title: "74. Search a 2D Matrix"
permalink: "/leetcode/problem/2023-08-07-74-search-a-2d-matrix/"
leetcode_ui: true
entry_slug: "2023-08-07-74-search-a-2d-matrix"
---

[74. Search a 2D Matrix](https://leetcode.com/problems/search-a-2d-matrix/description/) medium
[blog post](https://leetcode.com/problems/search-a-2d-matrix/solutions/3874453/kotlin-binary-search/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/07082023-74-search-a-2d-matrix?r=2bam17&utm_campaign=post&utm_medium=web)
![image.png](/assets/leetcode_daily_images/d86ca5b3.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/301

#### Problem TLDR

2D Binary Search

#### Intuition

Just a Binary Search

#### Approach

For more robust code:
* inclusive `lo` and `hi`
* the last condition `lo == hi`
* move borders `lo = mid + 1`, `hi = mid - 1`
* check the result
* use built-in functions

#### Complexity

- Time complexity:
$$O(log(n*m))$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun searchMatrix(matrix: Array<IntArray>, target: Int): Boolean {
        var lo = 0
        var hi = matrix.lastIndex
        while (lo <= hi) {
          val mid = lo + (hi - lo) / 2
          val row = matrix[mid]
          if (target in row.first()..row.last())
            return row.binarySearch(target) >= 0
          if (target < row.first()) hi = mid - 1 else lo = mid + 1
        }
        return false
    }

```

