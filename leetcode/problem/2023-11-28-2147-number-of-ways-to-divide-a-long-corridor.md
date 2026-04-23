---
layout: leetcode-entry
title: "2147. Number of Ways to Divide a Long Corridor"
permalink: "/leetcode/problem/2023-11-28-2147-number-of-ways-to-divide-a-long-corridor/"
leetcode_ui: true
entry_slug: "2023-11-28-2147-number-of-ways-to-divide-a-long-corridor"
---

[2147. Number of Ways to Divide a Long Corridor](https://leetcode.com/problems/number-of-ways-to-divide-a-long-corridor/description/) hard
[blog post](https://leetcode.com/problems/number-of-ways-to-divide-a-long-corridor/solutions/4337973/kotlin-cumulative-sum/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/28112023-2147-number-of-ways-to-divide?r=2bam17&utm_campaign=post&utm_medium=web)
[youtube](https://youtu.be/cbmDt5_-TSY)
![image.png](/assets/leetcode_daily_images/25df92ef.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/419

#### Problem TLDR

Count ways to place borders separating pairs of 'S' in 'SP' string

#### Intuition

We can scan linearly and do the interesting stuff after each two 'S': each new 'P' adds 'sum' ways to the total.
The last pair of 'S' don't need a border.

```
  // ssppspsppsspp
  // ss         1
  // ssp        2
  // sspp       3
  //     sps    3
  //     spsp   3+3=6
  //     spspp  6+3=9 <-- return this
  //           ss    9
  //           ssp   9+9=18
  //           sspp  18+9=27 discard this result, as it is last
```

#### Approach

Carefult what 'sum' to add, save the last sum to a separate variable.

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

  fun numberOfWays(corridor: String): Int {
    var prev = 1
    var sum = 1
    var s = 0
    for (c in corridor)
      if (c == 'S') {
        if (s == 2) {
          prev = sum
          s = 0
        }
        s++
      } else if (s == 2)
        sum = (prev + sum) % 1_000_000_007
    return if (s == 2) prev else 0
  }

```

