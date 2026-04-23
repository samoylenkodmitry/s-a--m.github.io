---
layout: leetcode-entry
title: "520. Detect Capital"
permalink: "/leetcode/problem/2023-01-02-520-detect-capital/"
leetcode_ui: true
entry_slug: "2023-01-02-520-detect-capital"
---

[520. Detect Capital](https://leetcode.com/problems/detect-capital/description/) easy

[https://t.me/leetcode_daily_unstoppable/72](https://t.me/leetcode_daily_unstoppable/72)

[blog post](https://leetcode.com/problems/detect-capital/solutions/2985088/kotlin-as-is/)

```kotlin
    fun detectCapitalUse(word: String): Boolean =
       word.all { Character.isUpperCase(it) } ||
       word.all { Character.isLowerCase(it) } ||
       Character.isUpperCase(word[0]) && word.drop(1).all { Character.isLowerCase(it) }

```

We can do this optimally by checking the first character and then checking all the other characters in a single pass. Or we can write a more understandable code that directly translates from the problem description.
Let's write one-liner.

Space: O(1), Time: O(N)

