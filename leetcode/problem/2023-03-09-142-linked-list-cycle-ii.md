---
layout: leetcode-entry
title: "142. Linked List Cycle II"
permalink: "/leetcode/problem/2023-03-09-142-linked-list-cycle-ii/"
leetcode_ui: true
entry_slug: "2023-03-09-142-linked-list-cycle-ii"
---

[142. Linked List Cycle II](https://leetcode.com/problems/linked-list-cycle-ii/description/) medium

[blog post](https://leetcode.com/problems/linked-list-cycle-ii/solutions/3275105/kotlin-fast-and-slow-plus-trick/)

```kotlin

fun detectCycle(head: ListNode?): ListNode? {
    var one = head
    var two = head
    do {
        one = one?.next
        two = two?.next?.next
    } while (two != null && one != two)
    if (two == null) return null
    one = head
    while (one != two) {
        one = one?.next
        two = two?.next
    }
    return one
}

```

#### Join me on telegram
https://t.me/leetcode_daily_unstoppable/143
#### Intuition
![image.png](/assets/leetcode_daily_images/01c1c6d1.webp)
There is a known algorithm to detect a cycle in a linked list. Move `slow` pointer one node at a time, and move `fast` pointer two nodes at a time. Eventually, if they meet, there is a cycle.
To know the connection point of the cycle, you can also use two pointers: one from where pointers were met, another from the start, and move both of them one node at a time until they meet.
How to derive this yourself?
* you can draw the diagram
* notice, when all the list is a cycle, nodes met at exactly where they are started
* meet point = cycle length + tail
#### Approach
* careful with corner cases.
#### Complexity
- Time complexity:
$$O(n)$$
- Space complexity:
$$O(1)$$

