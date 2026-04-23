---
layout: leetcode-entry
title: "92. Reverse Linked List II"
permalink: "/leetcode/problem/2023-09-07-92-reverse-linked-list-ii/"
leetcode_ui: true
entry_slug: "2023-09-07-92-reverse-linked-list-ii"
---

[92. Reverse Linked List II](https://leetcode.com/problems/reverse-linked-list-ii/description/) medium
[blog post](https://leetcode.com/problems/reverse-linked-list-ii/solutions/4012217/kotlin-dummy/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/7092023-92-reverse-linked-list-ii?utm_campaign=post&utm_medium=web)

![image.png](/assets/leetcode_daily_images/890f683b.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/332

#### Problem TLDR

Reverse a part of `Linked List`

#### Intuition

We need to find a point where to start reversing after `left` steps, then do the reversing `right - left` steps and finally connect to tail.

#### Approach

* use `Dummy head` technique to avoid reversed head corner case
* better do debug right in the code

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin
    fun reverseBetween(head: ListNode?, left: Int, right: Int): ListNode? {
      val dummy = ListNode(0).apply { next = head }
      var prev: ListNode? = dummy
      var curr = prev             // d-1-2-3-4-5  2 4
      repeat(left) {              // pc
        prev = curr               // p c
        curr = curr?.next         //   p c
      }
      val head = prev             // d-1-2-3-4-5  2 4
      val tail = curr             //   h t
      prev = curr
      curr = curr?.next           //     p c
      repeat(right - left) {      //     p c n
        val next = curr?.next     //      <p c n
        curr?.next = prev         //     p<c n
        prev = curr               //      <p<c n
        curr = next               //     2<p c
      }                           //     2<3<p c
      head?.next = prev           // d-1-2-3-4-5  2 4
      tail?.next = curr           //   h t<3<p c
      return dummy.next
    }

```

