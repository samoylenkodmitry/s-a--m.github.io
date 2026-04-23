---
layout: leetcode-entry
title: "1572. Matrix Diagonal Sum"
permalink: "/leetcode/problem/2023-05-08-1572-matrix-diagonal-sum/"
leetcode_ui: true
entry_slug: "2023-05-08-1572-matrix-diagonal-sum"
---

[1572. Matrix Diagonal Sum](https://leetcode.com/problems/matrix-diagonal-sum/description/) easy

```kotlin

fun diagonalSum(mat: Array<IntArray>): Int =
    (0..mat.lastIndex).sumBy {
        mat[it][it] + mat[it][mat.lastIndex - it]
    }!! - if (mat.size % 2 == 0) 0 else mat[mat.size / 2][mat.size / 2]

```

[blog post](https://leetcode.com/problems/matrix-diagonal-sum/solutions/3498716/kotlin-one-liner/)
[substack](https://dmitriisamoilenko.substack.com/p/leetcode-daily-8052023?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/206
#### Intuition
Just do what is asked.
#### Approach
* avoid double counting of the center element
#### Complexity
- Time complexity:
$$O(n)$$
- Space complexity:
$$O(1)$$

