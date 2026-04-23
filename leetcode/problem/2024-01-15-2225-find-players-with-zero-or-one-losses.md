---
layout: leetcode-entry
title: "2225. Find Players With Zero or One Losses"
permalink: "/leetcode/problem/2024-01-15-2225-find-players-with-zero-or-one-losses/"
leetcode_ui: true
entry_slug: "2024-01-15-2225-find-players-with-zero-or-one-losses"
---

[2225. Find Players With Zero or One Losses](https://leetcode.com/problems/find-players-with-zero-or-one-losses/description/) medium
[blog post](https://leetcode.com/problems/find-players-with-zero-or-one-losses/solutions/4567940/kotlin/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/15012024-2225-find-players-with-zero?r=2bam17&utm_campaign=post&utm_medium=web&showWelcome=true)
[youtube](https://youtu.be/SjZnYy5X244)
![image.png](/assets/leetcode_daily_images/1c07267f.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/472

#### Problem TLDR

[sorted winners list, sorted single lose list]

#### Intuition

No special algorithms here, just a `set` manipulation.

#### Approach

Let's use some Kotlin's API:
* map
* groupingBy
* eachCount
* filter
* sorted

#### Complexity

- Time complexity:
$$O(nlog(n))$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

  fun findWinners(matches: Array<IntArray>) = buildList {
    val winners = matches.map { it[0] }.toSet()
    val losers = matches.groupingBy { it[1] }.eachCount()
    add((winners - losers.keys).sorted())
    add(losers.filter { (k, v) -> v == 1 }.keys.sorted())
  }

```

