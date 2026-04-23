---
layout: leetcode-entry
title: "1721. Swapping Nodes in a Linked List"
permalink: "/leetcode/problem/2023-05-15-1721-swapping-nodes-in-a-linked-list/"
leetcode_ui: true
entry_slug: "2023-05-15-1721-swapping-nodes-in-a-linked-list"
---

[1721. Swapping Nodes in a Linked List](https://leetcode.com/problems/swapping-nodes-in-a-linked-list/description/) medium
[blog post](https://leetcode.com/problems/swapping-nodes-in-a-linked-list/solutions/3525455/kotlin-swap-values-not-nodes/)
[substack](https://dmitriisamoilenko.substack.com/p/15052023-1721-swapping-nodes-in-a?sd=pf)
#### Join me on Telegram
https://t.me/leetcode_daily_unstoppable/213
#### Problem TLDR
Swap the values of the head-tail k'th ListNodes.
#### Intuition
As we aren't asked to swap nodes, the problem is to find nodes.

#### Approach
Travel the `fast` pointer at `k` distance, then move both `fast` and `two` nodes until `fast` reaches the end.
#### Complexity
- Time complexity:
$$O(n)$$
- Space complexity:
$$O(1)$$

#### Code

```kotlin

fun swapNodes(head: ListNode?, k: Int): ListNode? {
    var fast = head
    for (i in 1..k - 1) fast = fast?.next
    val one = fast
    var two = head
    while (fast?.next != null) {
        two = two?.next
        fast = fast?.next
    }
    one?.`val` = two?.`val`.also { two?.`val` = one?.`val` }
    return head
}

```

