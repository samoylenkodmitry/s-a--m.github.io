---
layout: leetcode-entry
title: "19. Remove Nth Node From End of List"
permalink: "/leetcode/problem/2024-03-03-19-remove-nth-node-from-end-of-list/"
leetcode_ui: true
entry_slug: "2024-03-03-19-remove-nth-node-from-end-of-list"
---

[19. Remove Nth Node From End of List](https://leetcode.com/problems/remove-nth-node-from-end-of-list/description/) medium
[blog post](https://leetcode.com/problems/remove-nth-node-from-end-of-list/solutions/4814951/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/03032024-19-remove-nth-node-from?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/Iz7KFMO0-RU)
![image.png](/assets/leetcode_daily_images/97a158a4.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/527

#### Problem TLDR

Remove `n`th node from the tail of linked list.

#### Intuition

There is a two-pointer technique: fast pointer moves `n` nodes from the slow, then they go together until the end.
![image.png](/assets/leetcode_daily_images/335947ac.webp)

#### Approach

Some tricks:
* Use dummy first node to handle the head removal case.
* We can use counter to make it one pass.
Rust borrow checker makes the task non trivial: one pointer must be mutable, another must be cloned.

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

  fun removeNthFromEnd(head: ListNode?, n: Int): ListNode? {
    var r: ListNode = ListNode(0).apply { next = head }
    var a: ListNode? = r; var b: ListNode? = r; var i = 0
    while (b != null) { if (i++ > n) a = a?.next; b = b?.next }
    a?.next = a?.next?.next
    return r.next
  }

```
```rust

  pub fn remove_nth_from_end(head: Option<Box<ListNode>>, n: i32) -> Option<Box<ListNode>> {
    let mut r = ListNode { val: 0, next: head }; let mut r = Box::new(r);
    let mut b = r.clone(); let mut a = r.as_mut(); let mut i = 0;
    while b.next.is_some() {
      i+= 1; if i > n { a = a.next.as_mut().unwrap() }
      b = b.next.unwrap()
    }
    let n = a.next.as_mut().unwrap(); a.next = n.next.clone(); r.next
  }

```

