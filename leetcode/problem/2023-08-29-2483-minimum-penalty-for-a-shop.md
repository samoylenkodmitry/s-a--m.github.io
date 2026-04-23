---
layout: leetcode-entry
title: "2483. Minimum Penalty for a Shop"
permalink: "/leetcode/problem/2023-08-29-2483-minimum-penalty-for-a-shop/"
leetcode_ui: true
entry_slug: "2023-08-29-2483-minimum-penalty-for-a-shop"
---

[2483. Minimum Penalty for a Shop](https://leetcode.com/problems/minimum-penalty-for-a-shop/description/) medium
[blog post](https://leetcode.com/problems/minimum-penalty-for-a-shop/solutions/3974919/kotlin-closed-opened/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/29082023-2483-minimum-penalty-for?utm_campaign=post&utm_medium=web)
![image.png](/assets/leetcode_daily_images/c28024ed.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/323

#### Problem TLDR

First index of minimum penalty in array, penalty 'Y'-> 1, 'N' -> -1

#### Intuition
Iterate from the end and compute the suffix penalty.

#### Approach
Suffix penalty is a difference between `p_closed - p_opened`.

#### Complexity
- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun bestClosingTime(customers: String): Int {
      var p = 0
      var iMin = customers.length
      var pMin = 0
      for (i in customers.lastIndex downTo 0) {
        if (customers[i] == 'Y') p++ else p--
        if (p <= pMin) {
          iMin = i
          pMin = p
        }
      }
      return iMin
    }

```

