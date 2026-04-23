---
layout: leetcode-entry
title: "445. Add Two Numbers II"
permalink: "/leetcode/problem/2023-07-17-445-add-two-numbers-ii/"
leetcode_ui: true
entry_slug: "2023-07-17-445-add-two-numbers-ii"
---

[445. Add Two Numbers II](https://leetcode.com/problems/add-two-numbers-ii/description/) medium
[blog post](https://leetcode.com/problems/add-two-numbers-ii/solutions/3776193/kotlin/)
[substack](https://dmitriisamoilenko.substack.com/p/17072023-445-add-two-numbers-ii)
![image.png](/assets/leetcode_daily_images/7573faf4.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/278

#### Problem TLDR

Linked List of sum of two Linked Lists numbers, `9->9 + 1 = 1->0->0`

#### Intuition

The hint is in the description: reverse lists, then just do arithmetic. Another way is to use stack.

#### Approach

* don't forget to undo the reverse

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun addTwoNumbers(l1: ListNode?, l2: ListNode?, n: Int = 0): ListNode? {
      fun ListNode?.reverse(): ListNode? {
        var curr = this
        var prev: ListNode? = null
        while (curr != null) {
          val next = curr.next
          curr.next = prev
          prev = curr
          curr = next
        }
        return prev
      }
      var l1r = l1.reverse()
      var l2r = l2.reverse()
      var o = 0
      var prev: ListNode? = null
      while (l1r != null || l2r != null) {
        val v = o + (l1r?.`val` ?: 0) + (l2r?.`val` ?: 0)
        prev = ListNode(v % 10).apply { next = prev }
        o = v / 10
        l1r = l1r?.next
        l2r = l2r?.next
      }
      if (o > 0) prev = ListNode(o).apply { next = prev }
      l1r.reverse()
      l2r.reverse()
      return prev
    }

```

