---
layout: leetcode-entry
title: "2181. Merge Nodes in Between Zeros"
permalink: "/leetcode/problem/2024-07-04-2181-merge-nodes-in-between-zeros/"
leetcode_ui: true
entry_slug: "2024-07-04-2181-merge-nodes-in-between-zeros"
---

[2181. Merge Nodes in Between Zeros](https://leetcode.com/problems/merge-nodes-in-between-zeros/description/) medium
[blog post](https://leetcode.com/problems/merge-nodes-in-between-zeros/solutions/5413576/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/4072024-2181-merge-nodes-in-between?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/hWnzasAUV08)
![2024-07-04_09-23.webp](/assets/leetcode_daily_images/e665a041.webp)

https://t.me/leetcode_daily_unstoppable/659

#### Problem TLDR

Collapse in-between `0` nodes in a LinkedList #medium #linked_list

#### Intuition

Just do what is asked: iterate and modify the values and links on the fly.

#### Approach

* Kotlin: let's use just one extra variable
* Rust: I am sorry

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(1)$$

#### Code

```kotlin

    fun mergeNodes(head: ListNode?): ListNode? {
        var curr = head?.next
        while (curr?.next != null)
            if (curr.next?.`val` ?: 0 > 0) {
                curr.`val` += curr.next?.`val` ?: 0
                curr.next = curr.next?.next
            } else {
                curr.next = curr.next?.next
                curr = curr.next
            }
        return head?.next
    }

```
```rust

    pub fn merge_nodes(mut head: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
        let Some(head_box) = head.as_mut() else { return head };
        let mut curr = &mut head_box.next;
        while let Some(curr_box) = curr {
            let Some(next_box) = curr_box.next.as_mut() else { curr_box.next = None; break };
            if next_box.val > 0 {
                curr_box.val += next_box.val;
                curr_box.next = next_box.next.take()
            } else {
                curr_box.next = next_box.next.take();
                curr = &mut curr.as_mut().unwrap().next
            }
        }
        head.unwrap().next
    }

```

