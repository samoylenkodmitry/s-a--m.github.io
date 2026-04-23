---
layout: leetcode-entry
title: "1727. Largest Submatrix With Rearrangements"
permalink: "/leetcode/problem/2023-11-26-1727-largest-submatrix-with-rearrangements/"
leetcode_ui: true
entry_slug: "2023-11-26-1727-largest-submatrix-with-rearrangements"
---

[1727. Largest Submatrix With Rearrangements](https://leetcode.com/problems/largest-submatrix-with-rearrangements/description/) medium
[blog post](https://leetcode.com/problems/largest-submatrix-with-rearrangements/solutions/4330761/kotlin/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/26112023-1727-largest-submatrix-with?r=2bam17&utm_campaign=post&utm_medium=web)
[youtube](https://youtu.be/K-EQs20YOF0)
![image.png](/assets/leetcode_daily_images/01290387.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/417

#### Problem TLDR

Max area of `1` submatrix after sorting columns optimally

#### Intuition

Use hint :(
Ok, if we store the heights of the columns we can analyze each row independently, by choosing the largest heights first. The area will be `height * width`, where width will be the current position:
![image.png](/assets/leetcode_daily_images/6503c8f0.webp)

![image.png](/assets/leetcode_daily_images/0caba092.webp)

#### Approach

We can reuse the matrix, but don't do this in a production code without a warning.

#### Complexity

- Time complexity:
$$O(nmlog(m))$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

  fun largestSubmatrix(matrix: Array<IntArray>): Int {
    for (y in 1..<matrix.size)
      for (x in 0..<matrix[y].size)
        if (matrix[y][x] > 0)
          matrix[y][x] += matrix[y - 1][x]
    var max = 0
    for (row in matrix) {
      row.sort()
      for (x in row.lastIndex downTo 0)
        max = max(max, row[x] * (row.size - x))
    }
    return max
  }

```

