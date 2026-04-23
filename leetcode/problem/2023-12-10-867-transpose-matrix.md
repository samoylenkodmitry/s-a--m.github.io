---
layout: leetcode-entry
title: "867. Transpose Matrix"
permalink: "/leetcode/problem/2023-12-10-867-transpose-matrix/"
leetcode_ui: true
entry_slug: "2023-12-10-867-transpose-matrix"
---

[867. Transpose Matrix](https://leetcode.com/problems/transpose-matrix/description/) easy
[blog post](https://leetcode.com/problems/transpose-matrix/solutions/4385162/kotlin/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/10122023-867-transpose-matrix?r=2bam17&utm_campaign=post&utm_medium=web)
[youtube](https://youtu.be/nyXU1WVpcuo)
![image.png](/assets/leetcode_daily_images/bfd88747.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/434

#### Problem TLDR

Transpose 2D matrix

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

  fun transpose(matrix: Array<IntArray>): Array<IntArray> =
    Array(matrix[0].size) { x ->
      IntArray(matrix.size) { y ->
        matrix[y][x]
      }
    }

```

