---
layout: leetcode-entry
title: "859. Buddy Strings"
permalink: "/leetcode/problem/2023-07-03-859-buddy-strings/"
leetcode_ui: true
entry_slug: "2023-07-03-859-buddy-strings"
---

[859. Buddy Strings](https://leetcode.com/problems/buddy-strings/description/) easy
[blog post](https://leetcode.com/problems/buddy-strings/solutions/3710751/kotlin/)
[substack](https://dmitriisamoilenko.substack.com/p/3072023-859-buddy-strings?sd=pf)
![image.png](/assets/leetcode_daily_images/961756e2.webp)

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/264
#### Problem TLDR
Is it just one swap `s[i]<>s[j]` to string `s` == string `goal`
#### Intuition
Compare two strings for each position. There are must be only two not equal positions and they must be mirrored pairs.

#### Approach
Let's write it in Kotlin collections API style.

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

fun buddyStrings(s: String, goal: String): Boolean = s.length == goal.length && (
s == goal && s.groupBy { it }.any { it.value.size > 1 } ||
s.zip(goal)
.filter { (a, b) -> a != b }
.windowed(2)
.map { (ab, cd) -> listOf(ab, cd.second to cd.first) }
.let { it.size == 1 && it[0][0] == it[0][1] }
)

```

