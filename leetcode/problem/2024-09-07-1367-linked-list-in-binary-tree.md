---
layout: leetcode-entry
title: "1367. Linked List in Binary Tree"
permalink: "/leetcode/problem/2024-09-07-1367-linked-list-in-binary-tree/"
leetcode_ui: true
entry_slug: "2024-09-07-1367-linked-list-in-binary-tree"
---

[1367. Linked List in Binary Tree](https://leetcode.com/problems/linked-list-in-binary-tree/description/) medium
[blog post](https://leetcode.com/problems/linked-list-in-binary-tree/solutions/5749980/kotlin-rust/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/07092024-1367-linked-list-in-binary?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/NIIx34wsYMQ)
![1.webp](/assets/leetcode_daily_images/2ca10651.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/727

#### Problem TLDR

Is the LinkedList in the BinaryTree? #medium #linked_list #tree

#### Intuition

The problem size `n` is not that big, we can do a full Depth-First search and try to match Linked List at every tree node.

#### Approach

* the corner case is: `list: [1,2], tree: [1->1->2]`

#### Complexity

- Time complexity:
$$O(n^2)$$

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun isSubPath(head: ListNode?, root: TreeNode?, start: Boolean = false): Boolean =
        head == null || head.`val` == root?.`val` &&
        (isSubPath(head.next, root.left, true) || isSubPath(head.next, root.right, true))
        || root != null && !start && (isSubPath(head, root.left) || isSubPath(head, root.right))

```
```rust

    pub fn is_sub_path(head: Option<Box<ListNode>>, root: Option<Rc<RefCell<TreeNode>>>) -> bool {
        fn dfs(head: &Option<Box<ListNode>>, root: &Option<Rc<RefCell<TreeNode>>>, start: bool) -> bool {
            let Some(h) = head else { return true }; let Some(r) = root else { return false };
            let r = r.borrow();
            h.val == r.val && (dfs(&h.next, &r.left, true) || dfs(&h.next, &r.right, true))
            || !start && (dfs(head, &r.left, false) || dfs(head, &r.right, false))
        }
        dfs(&head, &root, false)
    }

```
```c++

    bool isSubPath(ListNode* head, TreeNode* root, bool start = 0) {
        return !head || root && root->val == head->val
        && (isSubPath(head->next, root->left, 1) || isSubPath(head->next, root->right, 1))
        || root && !start && (isSubPath(head, root->left) || isSubPath(head, root->right));
    }

```

