---
layout: leetcode-entry
title: "1732. Find the Highest Altitude"
permalink: "/leetcode/problem/2023-06-19-1732-find-the-highest-altitude/"
leetcode_ui: true
entry_slug: "2023-06-19-1732-find-the-highest-altitude"
---

[1732. Find the Highest Altitude](https://leetcode.com/problems/find-the-highest-altitude/description/) easy
[blog post](https://leetcode.com/problems/find-the-highest-altitude/solutions/3654634/kotlin-one-liner/)
[substack](https://dmitriisamoilenko.substack.com/p/18062023-1732-find-the-highest-altitude?sd=pf)
![image.png](/assets/leetcode_daily_images/73acb931.webp)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/250
#### Problem TLDR
Max running sum
#### Intuition
Just sum all the values and compute the `max`

#### Approach
Let's write Kotlin `fold` one-liner
#### Complexity
- Time complexity:
$$O(n)$$
- Space complexity:
$$O(1)$$

#### Code

```kotlin

fun largestAltitude(gain: IntArray): Int = gain
.fold(0 to 0) { (max, sum), t -> maxOf(max, sum + t) to (sum + t) }
.first

```

