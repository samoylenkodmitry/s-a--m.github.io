---
layout: leetcode-entry
title: "1669. Merge In Between Linked Lists"
permalink: "/leetcode/problem/2024-03-20-1669-merge-in-between-linked-lists/"
leetcode_ui: true
entry_slug: "2024-03-20-1669-merge-in-between-linked-lists"
---

[1669. Merge In Between Linked Lists](https://leetcode.com/problems/merge-in-between-linked-lists/description/) medium
[blog post](https://leetcode.com/problems/merge-in-between-linked-lists/solutions/4900331/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/20032024-1669-merge-in-between-linked?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/0NU6p7K7INY)
![2024-03-20_09-48.jpg](/assets/leetcode_daily_images/07b8bbd1.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/544

#### Problem TLDR

Replace a segment in a LinkedList #medium

#### Intuition

Just careful pointers iteration.

#### Approach

* use dummy to handle the first node removal
* better to write a separate cycles
* Rust is hard

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

  fun mergeInBetween(list1: ListNode?, a: Int, b: Int, list2: ListNode?) =
    ListNode(0).run {
      next = list1
      var curr: ListNode? = this
      for (i in 1..a) curr = curr?.next
      var after = curr?.next
      for (i in a..b) after = after?.next
      curr?.next = list2
      while (curr?.next != null) curr = curr?.next
      curr?.next = after
      next
    }

```
```rust

  pub fn merge_in_between(list1: Option<Box<ListNode>>, a: i32, b: i32, list2: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
    let mut dummy = Box::new(ListNode::new(0));
    dummy.next = list1;
    let mut curr = &mut dummy;
    for _ in 0..a { curr = curr.next.as_mut().unwrap() }
    let mut after = &mut curr.next;
    for _ in a..=b { after = &mut after.as_mut().unwrap().next }
    let after_b = after.take(); // Detach the rest of the list after `b`, this will allow the next line for the borrow checker
    curr.next = list2;
    while let Some(ref mut next) = curr.next { curr = next; }
    curr.next = after_b;
    dummy.next
  }

```

