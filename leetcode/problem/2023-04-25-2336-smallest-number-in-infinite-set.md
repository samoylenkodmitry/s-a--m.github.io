---
layout: leetcode-entry
title: "2336. Smallest Number in Infinite Set"
permalink: "/leetcode/problem/2023-04-25-2336-smallest-number-in-infinite-set/"
leetcode_ui: true
entry_slug: "2023-04-25-2336-smallest-number-in-infinite-set"
---

[2336. Smallest Number in Infinite Set](https://leetcode.com/problems/smallest-number-in-infinite-set/description/) medium

```kotlin

class SmallestInfiniteSet() {
    val links = IntArray(1001) { it + 1 }

    fun popSmallest(): Int {
        val smallest = links[0]
        val next = links[smallest]
        links[smallest] = 0
        links[0] = next
        return smallest
    }

    fun addBack(num: Int) {
        if (links[num] == 0) {
            var maxLink = 0
            while (links[maxLink] <= num) maxLink = links[maxLink]
            val next = links[maxLink]
            links[maxLink] = num
            links[num] = next
        }
    }

}

```

[blog post](https://leetcode.com/problems/smallest-number-in-infinite-set/solutions/3452738/kotlin-sparse-array/)
[substack](https://dmitriisamoilenko.substack.com/p/leetcode-daily-25042023?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/191
#### Intuition
Given the constraints, we can hold every element as a link node to another in an Array. This will give us $$O(1)$$ time for `pop` operation, but $$O(n)$$ for `addBack` in the worst case.
A more asymptotically optimal solution, is to use a `TreeSet` and a single pointer to the largest popped element.

#### Approach
Let's implement a sparse array.
##### Complexity
- Time complexity:
$$O(1)$$ - for `pop`
$$O(n)$$ - constructor and `addBack`
- Space complexity:
$$O(n)$$

