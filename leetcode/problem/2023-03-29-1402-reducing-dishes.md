---
layout: leetcode-entry
title: "1402. Reducing Dishes"
permalink: "/leetcode/problem/2023-03-29-1402-reducing-dishes/"
leetcode_ui: true
entry_slug: "2023-03-29-1402-reducing-dishes"
---

[1402. Reducing Dishes](https://leetcode.com/problems/reducing-dishes/submissions/924018548/) hard

[blog post](https://leetcode.com/problems/reducing-dishes/solutions/3354056/kotlin-nlogn/)

```kotlin

fun maxSatisfaction(satisfaction: IntArray): Int {
    satisfaction.sort()
    var max = 0
    var curr = 0
    var diff = 0
    for (i in satisfaction.lastIndex downTo 0) {
        diff += satisfaction[i]
        curr += diff
        max = maxOf(max, curr)
    }

    return max
}

```

#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/163
#### Intuition
Looking at the problem data examples, we intuitively deduce that the larger the number, the further it goes. We need to sort the array. With the negative numbers, we must compare all the results, excluding array prefixes.

#### Approach
The naive $$O(n^2)$$ solution will work. However, there is an optimal one if we simply go from the end.
#### Complexity
- Time complexity:
$$O(nlog_2(n))$$
- Space complexity:
$$O(n)$$

