---
layout: leetcode-entry
title: "86. Partition List"
permalink: "/leetcode/problem/2023-08-15-86-partition-list/"
leetcode_ui: true
entry_slug: "2023-08-15-86-partition-list"
---

[86. Partition List](https://leetcode.com/problems/partition-list/description/) medium
[blog post](https://leetcode.com/problems/partition-list/solutions/3911144/kotlin-dummies/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/15082023-86-partition-list?utm_campaign=post&utm_medium=web)

![image.png](/assets/leetcode_daily_images/1ded31f5.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/309

#### Problem TLDR

Partition a Linked List by `x` value

#### Intuition

Keep two nodes for `less` and for `more` than x, and add to them, iterating over the list. Finally, concatenate `more` to `less`.

#### Approach

* To avoid cycles, make sure to set each `next` to `null`
* Use `dummy head` technique

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun partition(head: ListNode?, x: Int): ListNode? {
        val dummyLess = ListNode(0)
        val dummyMore = ListNode(0)
        var curr = head
        var less = dummyLess
        var more = dummyMore
        while (curr != null) {
          if (curr.`val` < x) {
            less.next = curr
            less = curr
          } else {
            more.next = curr
            more = curr
          }
          val next = curr.next
          curr.next = null
          curr = next
        }
        less.next = dummyMore.next
        return dummyLess.next
    }

```

