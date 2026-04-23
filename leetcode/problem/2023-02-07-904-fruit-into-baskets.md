---
layout: leetcode-entry
title: "904. Fruit Into Baskets"
permalink: "/leetcode/problem/2023-02-07-904-fruit-into-baskets/"
leetcode_ui: true
entry_slug: "2023-02-07-904-fruit-into-baskets"
---

[904. Fruit Into Baskets](https://leetcode.com/problems/fruit-into-baskets/description/) medium

[blog post](https://leetcode.com/problems/fruit-into-baskets/solutions/3154719/kotlin-greedy/)

```kotlin
    fun totalFruit(fruits: IntArray): Int {
        if (fruits.size <= 2) return fruits.size
        var type1 = fruits[fruits.lastIndex]
        var type2 = fruits[fruits.lastIndex - 1]
        var count = 2
        var max = 2
        var prevType = type2
        var prevTypeCount = if (type1 == type2) 2 else 1
        for (i in fruits.lastIndex - 2 downTo 0) {
            val type = fruits[i]
            if (type == type1 || type == type2 || type1 == type2) {
                if (type1 == type2 && type != type1) type2 = type
                if (type == prevType) prevTypeCount++
                else prevTypeCount = 1
                count++
            } else {
                count = prevTypeCount + 1
                type2 = type
                type1 = prevType
                prevTypeCount = 1
            }
            max = maxOf(max, count)
            prevType = type
        }
        return max
    }

```

#### Join daily telegram
https://t.me/leetcode_daily_unstoppable/111
#### Intuition
We can scan fruits linearly from the tail and keep only two types of fruits.
#### Approach
* careful with corner cases
#### Complexity
- Time complexity:
  $$O(n)$$
- Space complexity:
  $$O(1)$$

