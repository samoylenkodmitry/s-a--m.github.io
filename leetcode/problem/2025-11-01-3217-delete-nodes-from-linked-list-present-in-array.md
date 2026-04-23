---
layout: leetcode-entry
title: "3217. Delete Nodes From Linked List Present in Array"
permalink: "/leetcode/problem/2025-11-01-3217-delete-nodes-from-linked-list-present-in-array/"
leetcode_ui: true
entry_slug: "2025-11-01-3217-delete-nodes-from-linked-list-present-in-array"
---

[3217. Delete Nodes From Linked List Present in Array](https://leetcode.com/problems/delete-nodes-from-linked-list-present-in-array/description/) medium
[blog post](https://leetcode.com/problems/delete-nodes-from-linked-list-present-in-array/solutions/7318117/kotlin-rust-by-samoylenkodmitry-gffr/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/01112025-3217-delete-nodes-from-linked?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/-pjCBDs8gX4)

![44f34127-6606-4b69-bcaf-e5ac536eace9 (1).webp](/assets/leetcode_daily_images/b14218ca.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/1160

#### Problem TLDR

Remove an array from linked list #medium #ll

#### Intuition

Convert the array to HashSet.

#### Approach

* use a dummy node in a case of a removal of the first node from LL
* use a nested while loop
* code can be rewritten to a single while loop

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin
// 47ms
    fun modifiedList(n: IntArray, h: ListNode?): ListNode? {
        val dummy = ListNode(0).apply { next = h }
        val s = n.toSet(); var curr = dummy
        while (curr.next != null)
            if (curr.next.`val` in s) curr.next = curr.next.next
            else curr = curr.next ?: break
        return dummy.next
    }

```
```rust
// 20ms
    pub fn modified_list(n: Vec<i32>, h: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
        let mut dummy = ListNode { val: 0, next: h };
        let mut cur = &mut dummy;  let mut s: HashSet<_> = n.iter().collect();
        while let Some(next_box) = cur.next.as_mut() {
            if s.contains(&next_box.val) {
                cur.next = next_box.next.take();;
            } else { cur = cur.next.as_mut().unwrap() }
        }
        dummy.next
    }

```

