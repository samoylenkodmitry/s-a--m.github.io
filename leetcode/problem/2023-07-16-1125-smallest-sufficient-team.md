---
layout: leetcode-entry
title: "1125. Smallest Sufficient Team"
permalink: "/leetcode/problem/2023-07-16-1125-smallest-sufficient-team/"
leetcode_ui: true
entry_slug: "2023-07-16-1125-smallest-sufficient-team"
---

[1125. Smallest Sufficient Team](https://leetcode.com/problems/smallest-sufficient-team/description/) hard
[blog post](https://leetcode.com/problems/smallest-sufficient-team/solutions/3771197/kotlin-dfs-memo/)
[substack](https://dmitriisamoilenko.substack.com/p/16072023-1125-smallest-sufficient?sd=pf)

![image.png](/assets/leetcode_daily_images/c2381bcc.webp)

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/277

#### Problem TLDR
Smallest `team` from `people with skills`, having all `required skills`

#### Intuition
The skills set size is less than `32`, so we can compute a `bitmask` for each of `people` and for the `required` skills.
Next, our task is to choose a set from `people` that result skills mask will be equal to the `required`.
We can do a full search, each time `skipping` or `adding` one mask from the `people`.
Observing the problem, we can see, that result is only depending on the current `mask` and all the `remaining` people. So, we can cache it.

#### Approach
* we can use a `HashMap` to store `skill to index`, but given a small set of skills, just do `indexOf` in O(60 * 16)
* add to the team in `post order`, as `dfs` must return only the result depending on the input arguments

#### Complexity

- Time complexity:
$$O(p2^s)$$, as full mask bits are 2^s, s - skills, p - people

- Space complexity:
$$O(p2^s)$$

#### Code

```kotlin

    fun smallestSufficientTeam(skills: Array<String>, people: List<List<String>>): IntArray {
        val peoplesMask = people.map {  it.fold(0) { r, t -> r or (1 shl skills.indexOf(t)) } }
        val cache = mutableMapOf<Pair<Int, Int>, List<Int>>()
        fun dfs(curr: Int, mask: Int): List<Int> =
          if (mask == (1 shl skills.size) - 1) listOf()
          else if (curr == people.size) people.indices.toList()
          else cache.getOrPut(curr to mask) {
            val skip = dfs(curr + 1, mask)
            val take = dfs(curr + 1, mask or peoplesMask[curr]) + curr
            if (skip.size < take.size) skip else take
          }
        return dfs(0, 0).toIntArray()
    }

```

