---
layout: leetcode-entry
title: "946. Validate Stack Sequences"
permalink: "/leetcode/problem/2023-04-13-946-validate-stack-sequences/"
leetcode_ui: true
entry_slug: "2023-04-13-946-validate-stack-sequences"
---

[946. Validate Stack Sequences](https://leetcode.com/problems/validate-stack-sequences/description/) medium

```kotlin

fun validateStackSequences(pushed: IntArray, popped: IntArray): Boolean =
with(Stack<Int>()) {
    var pop = 0
    pushed.forEach {
        push(it)
        while (isNotEmpty() && peek() == popped[pop]) {
            pop()
            pop++
        }
    }
    isEmpty()
}

```

[blog post](https://leetcode.com/problems/validate-stack-sequences/solutions/3411131/kotlin-stack/)
[substack](https://dmitriisamoilenko.substack.com/p/13042023?sd=pf)
#### Telegram
https://t.me/leetcode_daily_unstoppable/179
#### Intuition
Do simulation using a Stack.
#### Approach
* use one iteration and a second pointer for `pop`
* empty the stack after inserting an element
#### Complexity
- Time complexity:
$$O(n)$$
- Space complexity:
$$O(n)$$

