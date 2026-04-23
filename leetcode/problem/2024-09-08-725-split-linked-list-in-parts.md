---
layout: leetcode-entry
title: "725. Split Linked List in Parts"
permalink: "/leetcode/problem/2024-09-08-725-split-linked-list-in-parts/"
leetcode_ui: true
entry_slug: "2024-09-08-725-split-linked-list-in-parts"
---

[725. Split Linked List in Parts](https://leetcode.com/problems/split-linked-list-in-parts/description/) medium
[blog post](https://leetcode.com/problems/split-linked-list-in-parts/solutions/5754533/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/08092024-725-split-linked-list-in?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/epTmqqcpg3o)
![1.webp](/assets/leetcode_daily_images/a3833108.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/728

#### Problem TLDR

Split LinkedList into `k` parts #medium #linked_list

#### Intuition

This is a test of how clean your code can be.
Count first, split second.

#### Approach

* count in each bucket `i` is `n / k + (n % k > i)`
* Rust makes you feel helpless

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(k)$$

#### Code

```kotlin

    fun splitListToParts(head: ListNode?, k: Int): Array<ListNode?> {
        var n = 0; var curr = head
        while (curr != null) { n++; curr = curr.next }
        curr = head
        return Array(k) { i -> curr?.also {
                for (j in 2..(n / k + if (i < n % k) 1 else 0))
                    curr = curr?.next
                curr = curr?.next.also { curr?.next = null }
        }}
    }

```
```rust

    pub fn split_list_to_parts(mut head: Option<Box<ListNode>>, k: i32) -> Vec<Option<Box<ListNode>>> {
        let mut n = 0; let mut curr = &head;
        while let Some(curr_box) = curr { n += 1; curr = &curr_box.next }
        (0..k).map(|i| {
            let mut start = head.take();
            let mut x = &mut start;
            for j in 1..n / k + (n % k > i) as i32 {
                if let Some(x_box) = x { x = &mut x_box.next }
            }
            if let Some(x_box) = x { head = x_box.next.take() }
            start
        }).collect()
    }

```
```c++

    vector<ListNode*> splitListToParts(ListNode* head, int k) {
        int n = 0; ListNode* curr = head;
        while (curr) { n++; curr = curr->next; }
        vector<ListNode*> res;
        for (int i = 0; i < k; i++) {
            res.push_back(head);
            curr = head;
            for (int j = 1; j < n / k + (n % k > i); j++)
                curr = curr->next;
            if (curr) { head = curr->next; curr->next = NULL; }
        }
        return res;
    }

```

