---
layout: leetcode-entry
title: "2487. Remove Nodes From Linked List"
permalink: "/leetcode/problem/2024-05-06-2487-remove-nodes-from-linked-list/"
leetcode_ui: true
entry_slug: "2024-05-06-2487-remove-nodes-from-linked-list"
---

[2487. Remove Nodes From Linked List](https://leetcode.com/problems/remove-nodes-from-linked-list/description/) medium
[blog post](https://leetcode.com/problems/remove-nodes-from-linked-list/solutions/5119271/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/06052024-2487-remove-nodes-from-linked?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/mvPLXEbscGs)
![2024-05-06_09-06.webp](/assets/leetcode_daily_images/2a054ceb.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/594

#### Problem TLDR

Make a Linked List non-increasing #medium #linked_list

#### Intuition

The trivial way to solve it is to use a monotonic stack technique: remove from the stack all lesser nodes and always add the current.
However, there is a clever O(1) memory solution: just reverse the Linked List and iterate from the tail.

#### Approach

Let's save some lines of code just for the fun of it: can you use a single extra variable?

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun removeNodes(head: ListNode?): ListNode? {
        var m = head
        while (head?.next != null) {
            val next = head?.next?.next
            head?.next?.next = m
            m = head?.next
            head?.next = next
        }
        while (m != null) {
            val next = if (m == head) null else m.next
            if (m.`val` >= (head?.next?.`val` ?: 0)) {
                if (m == head) return head
                m.next = head?.next
                head?.next = m
            }
            m = next
        }
        return head?.next
    }

```
```rust

    pub fn remove_nodes(mut head: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
        let (mut curr, mut prev) = (head, None);
        while let Some(mut curr_box) = curr {
            let next = curr_box.next;
            curr_box.next = prev;
            prev = Some(curr_box);
            curr = next;
        }
        while let Some(mut prev_box) = prev {
            let next = prev_box.next;
            if prev_box.val >= curr.as_ref().map_or(0, |curr| curr.val) {
                prev_box.next = curr;
                curr = Some(prev_box);
            }
            prev = next
        }
        curr
    }

```

