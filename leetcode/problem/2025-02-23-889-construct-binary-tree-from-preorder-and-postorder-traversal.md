---
layout: leetcode-entry
title: "889. Construct Binary Tree from Preorder and Postorder Traversal"
permalink: "/leetcode/problem/2025-02-23-889-construct-binary-tree-from-preorder-and-postorder-traversal/"
leetcode_ui: true
entry_slug: "2025-02-23-889-construct-binary-tree-from-preorder-and-postorder-traversal"
---

[889. Construct Binary Tree from Preorder and Postorder Traversal](https://leetcode.com/problems/construct-binary-tree-from-preorder-and-postorder-traversal/description/) medium
[blog post](https://leetcode.com/problems/construct-binary-tree-from-preorder-and-postorder-traversal/solutions/6458423/kotlin-rust-by-samoylenkodmitry-lgki/)
[substack](https://open.substack.com/pub/dmitriisamoilenko/p/23022025-889-construct-binary-tree?r=2bam17&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
[youtube](https://youtu.be/E1zh0lo7pLo)
![1.webp](/assets/leetcode_daily_images/e5c3d3f9.webp)

#### Join me on Telegram

https://t.me/leetcode_daily_unstoppable/904

#### Problem TLDR

Tree from preorder & postorder #medium #stack

#### Intuition

Follow the preorder.
The tricky part is `null` detection:
* we can track that all values are to the left of the `postorder` index
* or, more clever from u/lee215/: when preorder meets postorder we are done in the current subtree

#### Approach

* we can slice arrays for subtrees; the interesting fact is preorder index as at most 2 positions right to the postorder and lengths are always equal
* Rust type evaluation is broken: it didn't see the `push` and stops on the first `get`

#### Complexity

- Time complexity:
$$O(n^2)$$ for index search ans slicing, O(n) for the pre[i] == post[i] check

- Space complexity:
$$O(n)$$

#### Code

```kotlin

    fun constructFromPrePost(pre: IntArray, post: IntArray): TreeNode? =
        if (pre.size < 1) null else TreeNode(pre[0]).apply { if (pre.size > 1) {
            val l = pre.size - 1; val j = post.indexOf(pre[1]) + 1;
            left = constructFromPrePost(pre.sliceArray(1..j), post.sliceArray(0..j))
            right = constructFromPrePost(pre.sliceArray(j + 1..l), post.sliceArray(j..<l))
        }}

```
```rust

    pub fn construct_from_pre_post(pre: Vec<i32>, post: Vec<i32>) -> Option<Rc<RefCell<TreeNode>>> {
        let mut s = vec![]; let mut j = 0;
        for v in pre {
            let n = Some(Rc::new(RefCell::new(TreeNode::new(v)))); if s.len() < 1 { s.push(n.clone()); continue }
            while s.last().and_then(|x| x.as_ref()).is_some_and(|x| x.borrow().val == post[j]) { s.pop(); j += 1 }
            if let Some(mut l) = s.last_mut().and_then(|x| x.as_mut()).map(|x| x.borrow_mut()){
                if l.left.is_none() { l.left = n.clone() } else { l.right = n.clone() }}
            s.push(n.clone());
        }; s[0].clone()
    }

```
```c++

    TreeNode* constructFromPrePost(vector<int>& pre, vector<int>& post, int* i = new int(0), int* j = new int(0)) {
        TreeNode* n = new TreeNode(pre[(*i)++]);
        if (n->val != post[*j]) n->left = constructFromPrePost(pre, post, i, j);
        if (n->val != post[*j]) n->right = constructFromPrePost(pre, post, i, j);
        (*j)++; return n;
    }

```

