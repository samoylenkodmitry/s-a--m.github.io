---
layout: leetcode-entry
title: "1582. Special Positions in a Binary Matrix"
permalink: "/leetcode/problem/2023-12-13-1582-special-positions-in-a-binary-matrix/"
leetcode_ui: true
entry_slug: "2023-12-13-1582-special-positions-in-a-binary-matrix"
---

[1582. Special Positions in a Binary Matrix](https://leetcode.com/problems/special-positions-in-a-binary-matrix/description/) easy
[blog post](https://leetcode.com/problems/special-positions-in-a-binary-matrix/solutions/4398174/kotlin/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/13122023-1582-special-positions-in?r=2bam17&utm_campaign=post&utm_medium=web)
![image.png](/assets/leetcode_daily_images/075c9679.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/437

#### Complexity

- Time complexity:
$$O((nm)^2)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun numSpecial(mat: Array<IntArray>): Int {
       var count = 0
       for (y in 0..mat.lastIndex)
        for (x in 0..mat[y].lastIndex)
          if (mat[y][x] == 1
            && (0..mat.lastIndex).filter { it != y }.all { mat[it][x] == 0}
            && (0..mat[y].lastIndex).filter { it != x }.all { mat[y][it] == 0})
              count++
       return count
    }

```

