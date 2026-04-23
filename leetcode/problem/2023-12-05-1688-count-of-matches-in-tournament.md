---
layout: leetcode-entry
title: "1688. Count of Matches in Tournament"
permalink: "/leetcode/problem/2023-12-05-1688-count-of-matches-in-tournament/"
leetcode_ui: true
entry_slug: "2023-12-05-1688-count-of-matches-in-tournament"
---

[1688. Count of Matches in Tournament](https://leetcode.com/problems/count-of-matches-in-tournament/description/) easy
[blog post](https://leetcode.com/problems/count-of-matches-in-tournament/solutions/4364363/kotlin/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/07122023-1688-count-of-matches-in?r=2bam17&utm_campaign=post&utm_medium=web)
[youtube](https://youtu.be/K_fMbBNu8N0)
![image.png](/assets/leetcode_daily_images/345869cb.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/428

#### Problem TLDR

Count of odd-even matches according to the rules `x/2` or `1+(x-1)/2`.

#### Intuition

The naive solution is to just implement what is asked.

#### Approach

Then you go read others people solutions and found this: `n-1`.

#### Complexity

- Time complexity:
$$O(log(n))$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

  fun numberOfMatches(n: Int): Int {
    var x = n
    var matches = 0
    while (x > 1) {
      if (x % 2 == 0) {
        matches += x / 2
        x = x / 2
      } else {
        matches += (x - 1) / 2
        x = 1 + (x - 1) / 2
      }
    }
    return matches
  }

```

