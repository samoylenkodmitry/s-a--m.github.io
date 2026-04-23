---
layout: leetcode-entry
title: "725. Split Linked List in Parts"
permalink: "/leetcode/problem/2023-09-06-725-split-linked-list-in-parts/"
leetcode_ui: true
entry_slug: "2023-09-06-725-split-linked-list-in-parts"
---

[725. Split Linked List in Parts](https://leetcode.com/problems/split-linked-list-in-parts/description/) medium
[blog post](https://leetcode.com/problems/split-linked-list-in-parts/solutions/4007931/kotlin-precompute-sizes/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/6092023-725-split-linked-list-in?utm_campaign=post&utm_medium=web)

![image.png](/assets/leetcode_daily_images/4902419e.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/331

#### Problem TLDR

Split `Linked List` into `k` almost equal lists

#### Intuition

First, precompute sizes, by adding to buckets one-by-one in a loop. Next, just move list pointer by sizes values.

#### Approach

Do not forget to disconnect nodes.

#### Complexity

 - Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$ for the sizes array and for the result

#### Code

```kotlin

    fun splitListToParts(head: ListNode?, k: Int): Array<ListNode?> {
      val sizes = IntArray(k)
      var i = 0
      var curr = head
      while (curr != null) {
        sizes[i++ % k]++
        curr = curr.next
      }
      curr = head
      return sizes.map { sz ->
        curr.also {
          repeat(sz - 1) { curr = curr?.next }
          curr = curr?.next.also { curr?.next = null }
        }
      }.toTypedArray()
    }

```

