---
layout: leetcode-entry
title: "206. Reverse Linked List"
permalink: "/leetcode/problem/2024-03-21-206-reverse-linked-list/"
leetcode_ui: true
entry_slug: "2024-03-21-206-reverse-linked-list"
---

[206. Reverse Linked List](https://leetcode.com/problems/reverse-linked-list/description/) easy
[blog post](https://leetcode.com/problems/reverse-linked-list/solutions/4904985/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/21032024-206-reverse-linked-list?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/d0DrIgmWtGQ)
![2024-03-21_09-47.jpg](/assets/leetcode_daily_images/fe344d9b.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/545

#### Problem TLDR

Reverse a Linked List #easy

#### Intuition

We need at least two pointers to store current node and previous.

#### Approach

In a recursive approach:
* treat result as a new head
* erase the link to the next
* next.next must point to the current

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$ or log(n) for the recursion

#### Code

```kotlin

  fun reverseList(head: ListNode?): ListNode? =
    head?.next?.let { next ->
      head.next = null
      reverseList(next).also { next?.next = head }
    } ?: head

```
```rust

  pub fn reverse_list(mut head: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
    let mut curr = head; let mut prev = None;
    while let Some(mut curr_box) = curr {
      let next = curr_box.next;
      curr_box.next = prev;
      prev = Some(curr_box);
      curr = next;
    }
    prev
  }

```

Bonus: just a single pointer solution

```Kotlin

  fun reverseList(head: ListNode?): ListNode? {
    var prev = head
    while (head?.next != null) {
      val next = head?.next?.next
      head?.next?.next = prev
      prev = head?.next
      head?.next = next
    }
    return prev
  }

```

![2024-03-21_13-02.jpg](/assets/leetcode_daily_images/1ecdfc52.webp)

