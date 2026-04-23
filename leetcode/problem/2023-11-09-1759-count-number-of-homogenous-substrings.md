---
layout: leetcode-entry
title: "1759. Count Number of Homogenous Substrings"
permalink: "/leetcode/problem/2023-11-09-1759-count-number-of-homogenous-substrings/"
leetcode_ui: true
entry_slug: "2023-11-09-1759-count-number-of-homogenous-substrings"
---

[1759. Count Number of Homogenous Substrings](https://leetcode.com/problems/count-number-of-homogenous-substrings/description/) medium
[blog post](https://leetcode.com/problems/count-number-of-homogenous-substrings/solutions/4267188/kotlin/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/09112023-1759-count-number-of-homogenous?r=2bam17&utm_campaign=post&utm_medium=web)
![image.png](/assets/leetcode_daily_images/f2fd93f8.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/398

#### Problem TLDR

Count of substrings of same chars

#### Intuition

Just count current len and add to total

```
  // abbcccaa   c t
  // a          1 1
  //  b         1 2
  //   b        2 4
  //    c       1 5
  //     c      2 7
  //      c     3 10
  //       a    1 11
  //        a   2 13
```

#### Approach

* don't forget to update `prev`

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

  fun countHomogenous(s: String): Int {
    var total = 0
    var count = 0
    var prev = '.'
    for (c in s) {
      if (c == prev) count++
      else count = 1
      total = (total + count) % 1_000_000_007
      prev = c
    }
    return total
  }

```

