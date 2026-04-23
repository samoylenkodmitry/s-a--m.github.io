---
layout: leetcode-entry
title: "3217. Delete Nodes From Linked List Present in Array"
permalink: "/leetcode/problem/2024-09-06-3217-delete-nodes-from-linked-list-present-in-array/"
leetcode_ui: true
entry_slug: "2024-09-06-3217-delete-nodes-from-linked-list-present-in-array"
---

[3217. Delete Nodes From Linked List Present in Array](https://leetcode.com/problems/delete-nodes-from-linked-list-present-in-array/description/) medium
[blog post](https://leetcode.com/problems/delete-nodes-from-linked-list-present-in-array/solutions/5745013/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/06092024-3217-delete-nodes-from-linked?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/GJgnE6XAtYc)
![1.webp](/assets/leetcode_daily_images/7e4159ff.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/726

#### Problem TLDR

Remove `nums` from a Linked List #medium #linked_list

#### Intuition

This is a test of how clean your code can be.

#### Approach

* use a `dummy` head to simplify the code
* in Rust it is challenging: better use `&` references to `ListNode` objects; use `take`

#### Complexity

- Time complexity:
$$O(n)$$

- Space complexity:
$$O(n)$$ for set

#### Code

```kotlin

    fun modifiedList(nums: IntArray, head: ListNode?): ListNode? {
        val dummy = ListNode(0).apply { next = head }
        val set = nums.toSet(); var curr = dummy
        while (curr.next != null)
            if (curr.next.`val` in set) curr.next = curr.next.next
            else curr = curr.next ?: break
        return dummy.next
    }

```
```rust

    pub fn modified_list(nums: Vec<i32>, head: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
        let set: HashSet<_> = nums.into_iter().collect();
        let mut dummy = ListNode { next: head, val: 0 };
        let mut curr = &mut dummy;
        while let Some(next_box) = curr.next.as_mut() {
            if set.contains(&next_box.val) {
                curr.next = next_box.next.take()
            } else {
                curr = curr.next.as_mut().unwrap()
            }
        }
        dummy.next
    }

```
```c++

    ListNode* modifiedList(vector<int>& nums, ListNode* head) {
        bitset<100001> set; for (int v: nums) set.set(v);
        ListNode* dummy = new ListNode(0); dummy->next = head;
        ListNode* curr = dummy;
        while (curr->next) if (set[curr->next->val])
            curr->next = curr->next->next;
        else curr = curr->next;
        return dummy->next;
    }

```

