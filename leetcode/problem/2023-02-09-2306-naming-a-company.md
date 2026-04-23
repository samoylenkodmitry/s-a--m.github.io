---
layout: leetcode-entry
title: "2306. Naming a Company"
permalink: "/leetcode/problem/2023-02-09-2306-naming-a-company/"
leetcode_ui: true
entry_slug: "2023-02-09-2306-naming-a-company"
---

[2306. Naming a Company](https://leetcode.com/problems/naming-a-company/description/) hard

[blog post](https://leetcode.com/problems/naming-a-company/solutions/3163405/kotlin-intersect-suffix-buckets/)

```kotlin
    fun distinctNames(ideas: Array<String>): Long {
        // c -> offee
        // d -> onuts
        // t -> ime, offee
        val prefToSuf = Array(27) { hashSetOf<String>() }
        for (idea in ideas)
            prefToSuf[idea[0].toInt() - 'a'.toInt()].add(idea.substring(1, idea.length))
        var count = 0L
        for (i in 0..26)
            for (j in i + 1..26)
                count += prefToSuf[i].count { !prefToSuf[j].contains(it) } * prefToSuf[j].count { ! prefToSuf[i].contains(it) }
        return count * 2L
    }

```

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/113
#### Intuition
If we group ideas by the suffixes and consider only the unique elements, the result will be the intersection of the sizes of the groups. (To deduce this you must sit and draw, or have a big brain, or just use a hint)

#### Approach
Group and multiply. Don't forget to remove repeating elements in each two groups.
#### Complexity
- Time complexity:
  $$O(26^2n)$$
- Space complexity:
  $$O(n)$$

