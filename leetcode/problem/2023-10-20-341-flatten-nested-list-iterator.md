---
layout: leetcode-entry
title: "341. Flatten Nested List Iterator"
permalink: "/leetcode/problem/2023-10-20-341-flatten-nested-list-iterator/"
leetcode_ui: true
entry_slug: "2023-10-20-341-flatten-nested-list-iterator"
---

[341. Flatten Nested List Iterator](https://leetcode.com/problems/flatten-nested-list-iterator/description/) medium
[blog post](https://leetcode.com/problems/flatten-nested-list-iterator/solutions/4188488/kotlin-stack/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/20102023-341-flatten-nested-list?r=2bam17&utm_campaign=post&utm_medium=web)
![image.png](/assets/leetcode_daily_images/4055aa73.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/376

#### Problem TLDR

Implement graph iterator

#### Intuition

We need to save all the deep levels positions, so let's use a Stack.

#### Approach

* we can store `nextInt` integer in a separate variable, or just leave it in a Stack and do `pop` on `next()`
* it is better to `advance` after each `next()` call to know if there is a next position
* careful with the order of elements when expanding

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

class NestedIterator(nestedList: List<NestedInteger>) : Stack<NestedInteger>() {
    init {
      addAll(nestedList.reversed())
      advance()
    }
    fun advance() {
      while (isNotEmpty() && !peek().isInteger()) {
        addAll(pop().list.reversed())
      }
    }
    fun next(): Int = pop().integer.also { advance() }
    fun hasNext(): Boolean = isNotEmpty()
}

```

