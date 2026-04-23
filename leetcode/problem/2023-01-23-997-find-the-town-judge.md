---
layout: leetcode-entry
title: "997. Find the Town Judge"
permalink: "/leetcode/problem/2023-01-23-997-find-the-town-judge/"
leetcode_ui: true
entry_slug: "2023-01-23-997-find-the-town-judge"
---

[997. Find the Town Judge](https://leetcode.com/problems/find-the-town-judge/description/) easy

[https://t.me/leetcode_daily_unstoppable/95](https://t.me/leetcode_daily_unstoppable/95)

[blog post](https://leetcode.com/problems/find-the-town-judge/solutions/3089245/kotlin-map-and-set/)

```kotlin
    fun findJudge(n: Int, trust: Array<IntArray>): Int {
        val judges = mutableMapOf<Int, MutableSet<Int>>()
        for (i in 1..n) judges[i] = mutableSetOf()
        val notJudges = mutableSetOf<Int>()
        trust.forEach { (from, judge) ->
            judges[judge]!! += from
            notJudges += from
        }
        judges.forEach { (judge, people) ->
            if (people.size == n - 1
                && !people.contains(judge)
                && !notJudges.contains(judge))
                return judge
        }
        return -1
    }

```

We need to count how much trust have each judge and also exclude all judges that have trust in someone.

* use map and set
* there is a better solution with just counting of trust, but it is not that clear to understand and prove

Space: O(max(N, T)), Time: O(max(N, T))

