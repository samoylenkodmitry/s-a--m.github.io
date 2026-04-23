---
layout: leetcode-entry
title: "24. Swap Nodes in Pairs"
permalink: "/leetcode/problem/2023-05-16-24-swap-nodes-in-pairs/"
leetcode_ui: true
entry_slug: "2023-05-16-24-swap-nodes-in-pairs"
---

[24. Swap Nodes in Pairs](https://leetcode.com/problems/swap-nodes-in-pairs/description/) medium
[blog post](https://leetcode.com/problems/swap-nodes-in-pairs/solutions/3529159/kotlin-be-explicit-to-avoid-bugs/)
[substack](https://dmitriisamoilenko.substack.com/p/16052023-24-swap-nodes-in-pairs?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/214
#### Problem TLDR
Swap adjacent ListNodes `a-b-c-d -> b-a-d-c`.
#### Intuition
Those kinds of problems are easy, but your task is to write it bug free from the first go.

#### Approach
For more robust code:
* use `dummy` head to track for a new head
* use explicit variables for each node in the configuration
* do debug code by writing down it values in the comments
#### Complexity
- Time complexity:
$$O(n)$$
- Space complexity:
$$O(1)$$

#### Code

```kotlin

fun swapPairs(head: ListNode?): ListNode? {
    val dummy = ListNode(0).apply { next = head }
    var curr: ListNode? = dummy
    while (curr?.next != null && curr?.next?.next != null) {
        // curr->one->two->next
        // curr->two->one->next
        var one = curr.next
        var two = one?.next
        val next = two?.next
        curr.next = two
        two?.next = one
        one?.next = next

        curr = one
    }
    return dummy.next
}

```

