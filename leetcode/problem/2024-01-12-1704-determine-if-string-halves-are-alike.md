---
layout: leetcode-entry
title: "1704. Determine if String Halves Are Alike"
permalink: "/leetcode/problem/2024-01-12-1704-determine-if-string-halves-are-alike/"
leetcode_ui: true
entry_slug: "2024-01-12-1704-determine-if-string-halves-are-alike"
---

[1704. Determine if String Halves Are Alike](https://leetcode.com/problems/determine-if-string-halves-are-alike/description/) easy
[blog post](https://leetcode.com/problems/determine-if-string-halves-are-alike/solutions/4550111/kotlin/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/12012024-1704-determine-if-string?r=2bam17&utm_campaign=post&utm_medium=web&showWelcome=true)
[youtube](https://youtu.be/TSdjY4YTRkc)
![image.png](/assets/leetcode_daily_images/31f7d396.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/469

#### Problem TLDR

https://t.me/leetcode_daily_unstoppable/469

#### Approach

Let's use some Kotlin's API:
* toSet
* take
* drop
* count

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$, can be O(1) with `asSequence`

#### Code

```kotlin

  val vw = "aeiouAEIOU".toSet()
  fun halvesAreAlike(s: String) =
    s.take(s.length / 2).count { it in vw } ==
    s.drop(s.length / 2).count { it in vw }

```

