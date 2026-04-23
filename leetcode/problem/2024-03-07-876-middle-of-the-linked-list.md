---
layout: leetcode-entry
title: "876. Middle of the Linked List"
permalink: "/leetcode/problem/2024-03-07-876-middle-of-the-linked-list/"
leetcode_ui: true
entry_slug: "2024-03-07-876-middle-of-the-linked-list"
---

[876. Middle of the Linked List](https://leetcode.com/problems/middle-of-the-linked-list/description/) easy
[blog post](https://leetcode.com/problems/middle-of-the-linked-list/solutions/4836061/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/07032024-876-middle-of-the-linked?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/oynybfr75wU)
![image.png](/assets/leetcode_daily_images/4fc3c5b0.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/531

#### Problem TLDR

Middle of the Linked List #easy

#### Intuition

Use Tortoise and Hare algorithm https://cp-algorithms.com/others/tortoise_and_hare.html

#### Approach

We can check `fast.next` or just `fast`, but careful with moving `slow`. Better test yourself with examples: `[1], [1,2], [1,2,3], [1,2,3,4]`.

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

  fun middleNode(head: ListNode?): ListNode? {
    var s = head; var f = s
    while (f?.next != null) {
      f = f?.next?.next; s = s?.next
    }
    return s
  }

```
```rust

    pub fn middle_node(head: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
      let (mut s, mut f) = (head.clone(), head);
      while f.is_some() {
        f = f.unwrap().next;
        if f.is_some() { f = f.unwrap().next; s = s.unwrap().next }
      }
      s
    }

```

